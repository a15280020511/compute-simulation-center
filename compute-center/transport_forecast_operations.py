#!/usr/bin/env python3
"""Bounded offline SUMO and StatsForecast operations."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compute_runner import ComputeError

MAX_NODES = 500
MAX_EDGES = 1000
MAX_ROUTES = 200
MAX_FLOWS = 500
MAX_SERIES = 100
MAX_OBSERVATIONS = 50000
MAX_HORIZON = 3650


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeError(f"{name} must be an array")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ComputeError(f"{name} must be between {minimum} and {maximum}")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _identifier(value: Any, name: str) -> str:
    text = str(value or "")
    if not text or len(text) > 64 or not all(ch.isalnum() or ch in "_.-" for ch in text):
        raise ComputeError(f"{name} is invalid")
    return text


def _sumo_home() -> Path:
    try:
        import sumo
    except ImportError as exc:
        raise ComputeError("eclipse-sumo optional dependency is not installed") from exc
    root = Path(str(sumo.SUMO_HOME))
    if not root.is_dir():
        raise ComputeError("SUMO_HOME is unavailable")
    return root


def _sumo_binary(root: Path, name: str) -> str:
    candidate = root / "bin" / name
    return str(candidate) if candidate.is_file() else name


def _run(command: list[str], timeout: int, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout, env={**os.environ, "SUMO_HOME": str(_sumo_home())})
    except subprocess.TimeoutExpired as exc:
        raise ComputeError("SUMO execution timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise ComputeError((exc.stderr or exc.stdout or "SUMO execution failed")[-2000:]) from exc


def sumo_micro_simulation(inputs: Mapping[str, Any]) -> dict[str, Any]:
    nodes_raw = _sequence(inputs.get("nodes"), "inputs.nodes")
    edges_raw = _sequence(inputs.get("edges"), "inputs.edges")
    routes_raw = _sequence(inputs.get("routes"), "inputs.routes")
    flows_raw = _sequence(inputs.get("flows"), "inputs.flows")
    if not 2 <= len(nodes_raw) <= MAX_NODES: raise ComputeError(f"nodes must contain 2 to {MAX_NODES} entries")
    if not 1 <= len(edges_raw) <= MAX_EDGES: raise ComputeError(f"edges must contain 1 to {MAX_EDGES} entries")
    if not 1 <= len(routes_raw) <= MAX_ROUTES: raise ComputeError(f"routes must contain 1 to {MAX_ROUTES} entries")
    if not 1 <= len(flows_raw) <= MAX_FLOWS: raise ComputeError(f"flows must contain 1 to {MAX_FLOWS} entries")
    duration = _integer(inputs.get("duration_seconds", 3600), "inputs.duration_seconds", 60, 86400)
    seed = _integer(inputs.get("seed", 42), "inputs.seed", 1, 2147483647)
    timeout = _integer(inputs.get("timeout_seconds", 60), "inputs.timeout_seconds", 5, 120)

    nodes=[]; node_ids=set()
    for i, raw in enumerate(nodes_raw):
        row=_mapping(raw,f"inputs.nodes[{i}]"); node_id=_identifier(row.get("id"),f"nodes[{i}].id")
        if node_id in node_ids: raise ComputeError("node ids must be unique")
        node_ids.add(node_id); nodes.append({"id":node_id,"x":_number(row.get("x"),f"nodes[{i}].x",-1e9,1e9),"y":_number(row.get("y"),f"nodes[{i}].y",-1e9,1e9),"type":str(row.get("type") or "priority")})

    edges=[]; edge_ids=set()
    for i, raw in enumerate(edges_raw):
        row=_mapping(raw,f"inputs.edges[{i}]"); edge_id=_identifier(row.get("id"),f"edges[{i}].id"); source=_identifier(row.get("from"),f"edges[{i}].from"); target=_identifier(row.get("to"),f"edges[{i}].to")
        if edge_id in edge_ids or source not in node_ids or target not in node_ids or source==target: raise ComputeError("edges require unique ids and valid distinct endpoint nodes")
        edge_ids.add(edge_id); edges.append({"id":edge_id,"from":source,"to":target,"numLanes":str(_integer(row.get("lanes",1),f"edges[{i}].lanes",1,12)),"speed":str(_number(row.get("speed_mps",13.89),f"edges[{i}].speed_mps",0.1,80.0))})

    routes=[]; route_ids=set()
    for i, raw in enumerate(routes_raw):
        row=_mapping(raw,f"inputs.routes[{i}]"); route_id=_identifier(row.get("id"),f"routes[{i}].id"); route_edges=[_identifier(item,f"routes[{i}].edges") for item in _sequence(row.get("edges"),f"routes[{i}].edges")]
        if route_id in route_ids or not route_edges or any(item not in edge_ids for item in route_edges): raise ComputeError("routes require unique ids and registered edge ids")
        route_ids.add(route_id); routes.append({"id":route_id,"edges":" ".join(route_edges)})

    flows=[]
    for i, raw in enumerate(flows_raw):
        row=_mapping(raw,f"inputs.flows[{i}]"); flow_id=_identifier(row.get("id"),f"flows[{i}].id"); route_id=_identifier(row.get("route"),f"flows[{i}].route")
        if route_id not in route_ids: raise ComputeError("flow route is not registered")
        begin=_number(row.get("begin",0),f"flows[{i}].begin",0,duration); end=_number(row.get("end",duration),f"flows[{i}].end",begin,duration); rate=_number(row.get("vehicles_per_hour"),f"flows[{i}].vehicles_per_hour",0.1,20000)
        flows.append({"id":flow_id,"route":route_id,"begin":str(begin),"end":str(end),"vehsPerHour":str(rate),"departLane":"best","departSpeed":"max"})

    root=_sumo_home()
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp); nodes_xml=ET.Element("nodes")
        for row in nodes: ET.SubElement(nodes_xml,"node",{k:str(v) for k,v in row.items()})
        ET.ElementTree(nodes_xml).write(work/"nodes.nod.xml",encoding="utf-8",xml_declaration=True)
        edges_xml=ET.Element("edges")
        for row in edges: ET.SubElement(edges_xml,"edge",row)
        ET.ElementTree(edges_xml).write(work/"edges.edg.xml",encoding="utf-8",xml_declaration=True)
        routes_xml=ET.Element("routes"); ET.SubElement(routes_xml,"vType",{"id":"passenger","vClass":"passenger","accel":"2.6","decel":"4.5","length":"5","maxSpeed":"55"})
        for row in routes: ET.SubElement(routes_xml,"route",row)
        for row in flows: ET.SubElement(routes_xml,"flow",{**row,"type":"passenger"})
        ET.ElementTree(routes_xml).write(work/"routes.rou.xml",encoding="utf-8",xml_declaration=True)
        net_file=work/"network.net.xml"
        _run([_sumo_binary(root,"netconvert"),"--node-files","nodes.nod.xml","--edge-files","edges.edg.xml","--output-file",str(net_file),"--no-turnarounds","true"],timeout,work)
        trip_file=work/"tripinfo.xml"
        result=_run([_sumo_binary(root,"sumo"),"--net-file",str(net_file),"--route-files","routes.rou.xml","--begin","0","--end",str(duration),"--seed",str(seed),"--tripinfo-output",str(trip_file),"--no-step-log","true","--duration-log.disable","true","--no-warnings","true"],timeout,work)
        trips=list(ET.parse(trip_file).getroot().findall("tripinfo"))
        values={name:[float(row.attrib.get(name,0)) for row in trips] for name in ("duration","routeLength","waitingTime","timeLoss")}
        mean=lambda name: float(sum(values[name])/len(values[name])) if values[name] else 0.0
        return {"mode":"sumo_micro_simulation","sumo_version":"1.27.1","seed":seed,"duration_seconds":duration,"network":{"nodes":len(nodes),"edges":len(edges),"routes":len(routes),"flows":len(flows)},"completed_trips":len(trips),"mean_trip_duration_seconds":mean("duration"),"mean_route_length_meters":mean("routeLength"),"mean_waiting_time_seconds":mean("waitingTime"),"mean_time_loss_seconds":mean("timeLoss"),"network_policy":"deny","arbitrary_commands_allowed":False,"arbitrary_paths_allowed":False,"decision_support_only":True}


def statsforecast_batch(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive
    except ImportError as exc:
        raise ComputeError("statsforecast optional dependency is not installed") from exc
    rows=_sequence(inputs.get("series"),"inputs.series")
    if not 3<=len(rows)<=MAX_OBSERVATIONS: raise ComputeError(f"series must contain 3 to {MAX_OBSERVATIONS} rows")
    frame_rows=[]; identifiers=set()
    for i, raw in enumerate(rows):
        row=_mapping(raw,f"inputs.series[{i}]"); uid=_identifier(row.get("unique_id"),f"series[{i}].unique_id"); value=_number(row.get("y"),f"series[{i}].y",-1e15,1e15); date=str(row.get("ds") or "")
        if not date or len(date)>40: raise ComputeError("series ds must be an ISO-like date/time string")
        frame_rows.append({"unique_id":uid,"ds":date,"y":value}); identifiers.add(uid)
    if not 1<=len(identifiers)<=MAX_SERIES: raise ComputeError(f"series may contain at most {MAX_SERIES} unique ids")
    frequency=str(inputs.get("frequency") or "D")
    if frequency not in {"D","B","W","MS","ME","QS","QE","YS","YE","h"}: raise ComputeError("frequency is not allowlisted")
    horizon=_integer(inputs.get("horizon"),"inputs.horizon",1,MAX_HORIZON); season_length=_integer(inputs.get("season_length",1),"inputs.season_length",1,8760)
    raw_models=[str(item) for item in _sequence(inputs.get("models",["Naive","AutoETS"]),"inputs.models")]
    if not 1<=len(raw_models)<=5 or len(set(raw_models))!=len(raw_models): raise ComputeError("models must contain 1 to 5 unique allowlisted names")
    factories={"Naive":lambda:Naive(),"SeasonalNaive":lambda:SeasonalNaive(season_length=season_length),"AutoARIMA":lambda:AutoARIMA(season_length=season_length),"AutoETS":lambda:AutoETS(season_length=season_length),"AutoTheta":lambda:AutoTheta(season_length=season_length)}
    unknown=sorted(set(raw_models)-set(factories))
    if unknown: raise ComputeError(f"unsupported models: {unknown}")
    levels=[_integer(item,"inputs.levels",1,99) for item in _sequence(inputs.get("levels",[80,95]),"inputs.levels")]
    if len(levels)>3 or len(set(levels))!=len(levels): raise ComputeError("levels must contain at most 3 unique values")
    df=pd.DataFrame(frame_rows); df["ds"]=pd.to_datetime(df["ds"],errors="raise"); df=df.sort_values(["unique_id","ds"])
    if df.duplicated(["unique_id","ds"]).any(): raise ComputeError("series contains duplicate unique_id/ds rows")
    if int(df.groupby("unique_id").size().min())<max(3,season_length+1): raise ComputeError("each series must contain enough observations for the season length")
    sf=StatsForecast(models=[factories[name]() for name in raw_models],freq=frequency,n_jobs=1,fallback_model=Naive()); forecast=sf.forecast(df=df,h=horizon,level=levels)
    forecast["ds"]=forecast["ds"].astype(str)
    records=forecast.where(forecast.notna(),None).to_dict(orient="records")
    return {"mode":"statsforecast_batch","library":"statsforecast","series_count":len(identifiers),"observation_count":len(df),"horizon":horizon,"frequency":frequency,"season_length":season_length,"models":raw_models,"levels":levels,"forecast_rows":records,"network_policy":"deny","model_calls":0,"decision_support_only":True}


def transport_forecast_analysis(inputs: Mapping[str, Any]) -> dict[str, Any]:
    mode=str(inputs.get("mode") or "")
    if mode=="sumo_micro_simulation": return sumo_micro_simulation(inputs)
    if mode=="statsforecast_batch": return statsforecast_batch(inputs)
    raise ComputeError("inputs.mode must be sumo_micro_simulation or statsforecast_batch")

OPERATIONS={"transport_forecast_analysis":transport_forecast_analysis}
