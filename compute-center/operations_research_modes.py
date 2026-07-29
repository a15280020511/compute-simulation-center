#!/usr/bin/env python3
"""Bounded operations-research modes backed by Google OR-Tools."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from compute_runner import ComputeError

MAX_VARIABLES = 200
MAX_CONSTRAINTS = 1000
MAX_ASSIGNMENT_SIZE = 100
MAX_ROUTING_NODES = 200
MAX_VEHICLES = 20


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _solver():
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise ComputeError("operations-research optional dependency OR-Tools is not installed") from exc
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise ComputeError("OR-Tools SCIP solver is unavailable")
    return solver, pywraplp


def mixed_integer_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    solver, pywraplp = _solver()
    raw_variables = _sequence(inputs.get("variables"), "inputs.variables")
    if not 1 <= len(raw_variables) <= MAX_VARIABLES:
        raise ComputeError(f"inputs.variables must contain 1 to {MAX_VARIABLES} entries")
    variables = []
    names: list[str] = []
    objective_coefficients: list[float] = []
    for index, raw in enumerate(raw_variables):
        row = _mapping(raw, f"inputs.variables[{index}]")
        name = str(row.get("name") or "")
        kind = str(row.get("type") or "continuous")
        lower_value = row.get("lower_bound", row.get("lower", 0.0))
        upper_value = row.get("upper_bound", row.get("upper"))
        objective_value = row.get("objective_coefficient", row.get("objective", 0.0))
        lower = _finite(lower_value, f"variable[{name}].lower_bound")
        upper = solver.infinity() if upper_value is None else _finite(upper_value, f"variable[{name}].upper_bound")
        objective = _finite(objective_value, f"variable[{name}].objective_coefficient")
        if not name or name in names or lower > upper:
            raise ComputeError("variable names must be unique and bounds valid")
        if kind == "continuous":
            variable = solver.NumVar(lower, upper, name)
        elif kind == "integer":
            variable = solver.IntVar(math.ceil(lower), math.floor(upper), name)
        elif kind == "binary":
            if lower > 1 or upper < 0:
                raise ComputeError("binary variable bounds exclude both 0 and 1")
            variable = solver.BoolVar(name)
        else:
            raise ComputeError("variable type must be continuous, integer, or binary")
        variables.append(variable)
        names.append(name)
        objective_coefficients.append(objective)
    raw_constraints = _sequence(inputs.get("constraints", []), "inputs.constraints")
    if len(raw_constraints) > MAX_CONSTRAINTS:
        raise ComputeError(f"inputs.constraints must contain at most {MAX_CONSTRAINTS} entries")
    name_to_index = {name: i for i, name in enumerate(names)}
    for index, raw in enumerate(raw_constraints):
        row = _mapping(raw, f"inputs.constraints[{index}]")
        coefficients = _mapping(row.get("coefficients"), f"constraint[{index}].coefficients")
        relation = str(row.get("relation") or "<=")
        rhs = _finite(row.get("rhs"), f"constraint[{index}].rhs")
        unknown = set(coefficients) - set(names)
        if unknown:
            raise ComputeError(f"constraint references unknown variables: {sorted(unknown)}")
        if relation == "<=":
            constraint = solver.Constraint(-solver.infinity(), rhs)
        elif relation == ">=":
            constraint = solver.Constraint(rhs, solver.infinity())
        elif relation == "==":
            constraint = solver.Constraint(rhs, rhs)
        else:
            raise ComputeError("constraint relation must be <=, >=, or ==")
        for name, coefficient in coefficients.items():
            constraint.SetCoefficient(variables[name_to_index[str(name)]], _finite(coefficient, f"constraint[{index}].{name}"))
    objective = solver.Objective()
    for variable, coefficient in zip(variables, objective_coefficients):
        objective.SetCoefficient(variable, coefficient)
    maximize = bool(inputs.get("maximize", True))
    objective.SetMaximization() if maximize else objective.SetMinimization()
    time_limit = _integer(inputs.get("time_limit_seconds", 20), "inputs.time_limit_seconds", 1, 120)
    solver.SetTimeLimit(time_limit * 1000)
    status = solver.Solve()
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        status_names = {
            pywraplp.Solver.INFEASIBLE: "infeasible",
            pywraplp.Solver.UNBOUNDED: "unbounded",
            pywraplp.Solver.ABNORMAL: "abnormal",
            pywraplp.Solver.MODEL_INVALID: "model_invalid",
            pywraplp.Solver.NOT_SOLVED: "not_solved",
        }
        raise ComputeError(f"mixed-integer optimization failed: {status_names.get(status, status)}")
    return {
        "mode": "mixed_integer_optimization",
        "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible_time_limited",
        "objective_value": float(objective.Value()),
        "maximize": maximize,
        "variables": {names[i]: float(variables[i].solution_value()) for i in range(len(names))},
        "best_bound": float(objective.BestBound()),
        "wall_time_milliseconds": int(solver.wall_time()),
        "iterations": int(solver.iterations()),
        "nodes": int(solver.nodes()),
        "constraint_count": len(raw_constraints),
        "optimality_not_guaranteed": status != pywraplp.Solver.OPTIMAL,
        "decision_support_only": True,
    }


def assignment_optimization(inputs: Mapping[str, Any]) -> dict[str, Any]:
    solver, pywraplp = _solver()
    workers = [str(item) for item in _sequence(inputs.get("workers"), "inputs.workers")]
    tasks = [str(item) for item in _sequence(inputs.get("tasks"), "inputs.tasks")]
    if not 1 <= len(workers) <= MAX_ASSIGNMENT_SIZE or not 1 <= len(tasks) <= MAX_ASSIGNMENT_SIZE:
        raise ComputeError(f"workers and tasks must contain 1 to {MAX_ASSIGNMENT_SIZE} entries")
    if any(not item for item in workers + tasks) or len(set(workers)) != len(workers) or len(set(tasks)) != len(tasks):
        raise ComputeError("worker and task names must be non-empty and unique")
    costs = np.asarray([
        [_finite(value, f"inputs.costs[{i}][{j}]") for j, value in enumerate(_sequence(row, f"inputs.costs[{i}]"))]
        for i, row in enumerate(_sequence(inputs.get("costs"), "inputs.costs"))
    ], dtype=float)
    if costs.shape != (len(workers), len(tasks)):
        raise ComputeError("costs must be a worker-by-task matrix")
    maximize = bool(inputs.get("maximize", False))
    variables = {(i, j): solver.BoolVar(f"assign_{i}_{j}") for i in range(len(workers)) for j in range(len(tasks))}
    for i in range(len(workers)):
        solver.Add(sum(variables[i, j] for j in range(len(tasks))) <= 1)
    require_all_tasks = bool(inputs.get("require_all_tasks", True))
    for j in range(len(tasks)):
        expression = sum(variables[i, j] for i in range(len(workers)))
        solver.Add(expression == 1 if require_all_tasks else expression <= 1)
    objective = solver.Objective()
    for (i, j), variable in variables.items():
        objective.SetCoefficient(variable, float(costs[i, j]))
    objective.SetMaximization() if maximize else objective.SetMinimization()
    status = solver.Solve()
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise ComputeError("assignment optimization is infeasible")
    assignments = [
        {"worker": workers[i], "task": tasks[j], "value": float(costs[i, j])}
        for (i, j), variable in variables.items() if variable.solution_value() > 0.5
    ]
    return {
        "mode": "assignment_optimization",
        "objective_value": float(objective.Value()),
        "maximize": maximize,
        "assignments": assignments,
        "unassigned_workers": sorted(set(workers) - {row["worker"] for row in assignments}),
        "unassigned_tasks": sorted(set(tasks) - {row["task"] for row in assignments}),
        "decision_support_only": True,
    }


def vehicle_routing(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise ComputeError("operations-research optional dependency OR-Tools is not installed") from exc
    matrix = np.asarray([
        [_integer(value, f"inputs.distance_matrix[{i}][{j}]", 0, 10**9) for j, value in enumerate(_sequence(row, f"inputs.distance_matrix[{i}]"))]
        for i, row in enumerate(_sequence(inputs.get("distance_matrix"), "inputs.distance_matrix"))
    ], dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not 2 <= matrix.shape[0] <= MAX_ROUTING_NODES:
        raise ComputeError(f"distance_matrix must be square with 2 to {MAX_ROUTING_NODES} nodes")
    vehicle_count = _integer(inputs.get("vehicle_count"), "inputs.vehicle_count", 1, min(MAX_VEHICLES, matrix.shape[0] - 1))
    depot = _integer(inputs.get("depot", 0), "inputs.depot", 0, matrix.shape[0] - 1)
    manager = pywrapcp.RoutingIndexManager(matrix.shape[0], vehicle_count, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return int(matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    demands_raw = inputs.get("demands")
    capacities_raw = inputs.get("vehicle_capacities")
    demands: list[int] | None = None
    if demands_raw is not None or capacities_raw is not None:
        demands = [_integer(value, f"inputs.demands[{i}]", 0, 10**9) for i, value in enumerate(_sequence(demands_raw, "inputs.demands"))]
        capacities = [_integer(value, f"inputs.vehicle_capacities[{i}]", 1, 10**9) for i, value in enumerate(_sequence(capacities_raw, "inputs.vehicle_capacities"))]
        if len(demands) != matrix.shape[0] or len(capacities) != vehicle_count:
            raise ComputeError("demands must match nodes and vehicle_capacities vehicles")

        def demand_callback(index: int) -> int:
            return demands[manager.IndexToNode(index)]

        demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_index, 0, capacities, True, "Capacity")
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = _integer(inputs.get("time_limit_seconds", 10), "inputs.time_limit_seconds", 1, 120)
    search.log_search = False
    solution = routing.SolveWithParameters(search)
    if solution is None:
        raise ComputeError("vehicle routing problem is infeasible or timed out without a solution")
    routes = []
    total_distance = 0
    for vehicle in range(vehicle_count):
        index = routing.Start(vehicle)
        route = []
        distance = 0
        load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            if demands is not None:
                load += demands[node]
            next_index = solution.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(index, next_index, vehicle)
            index = next_index
        route.append(manager.IndexToNode(index))
        total_distance += distance
        routes.append({"vehicle": vehicle, "route": route, "distance": int(distance), "load": int(load)})
    return {
        "mode": "vehicle_routing",
        "solver_status": int(routing.status()),
        "vehicle_count": vehicle_count,
        "depot": depot,
        "total_distance": int(total_distance),
        "routes": routes,
        "optimality_not_guaranteed": True,
        "decision_support_only": True,
    }


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "mixed_integer_optimization": mixed_integer_optimization,
    "assignment_optimization": assignment_optimization,
    "vehicle_routing": vehicle_routing,
}
