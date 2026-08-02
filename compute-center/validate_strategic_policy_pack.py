#!/usr/bin/env python3
"""Fixed local-data validation for strategic policy intelligence modes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from strategic_policy_intelligence_operations import HANDLERS

FIXTURES = {'axelrod_strategy_tournament': {'seed': 7,
                                 'strategies': ['Cooperator', 'Defector', 'Tit For Tat'],
                                 'turns': 20},
 'biogeme_choice_share': {'alternatives': [{'name': 'A', 'utility': 1},
                                           {'name': 'B', 'utility': 0.5},
                                           {'name': 'C', 'utility': -0.2}]},
 'claim_evidence_contradiction': {'claims': ['C1', 'C2'],
                                  'evidence': [{'claim': 'C1', 'stance': 'support', 'weight': 2},
                                               {'claim': 'C1', 'stance': 'contradict', 'weight': 1},
                                               {'claim': 'C2', 'stance': 'neutral', 'weight': 1}]},
 'clingo_rule_action_set': {'actions': ['invest', 'partner', 'exit'],
                            'forbidden_pairs': [['invest', 'exit']],
                            'max_selected': 2,
                            'max_solutions': 20,
                            'min_selected': 1,
                            'required': []},
 'datasketch_set_similarity': {'num_perm': 32,
                               'sets': {'a': ['x', 'y', 'z'], 'b': ['x', 'y'], 'c': ['q', 'r']}},
 'event_timeline_collision': {'events': [{'end': 3, 'entity': 'X', 'id': 'E1', 'location': 'A', 'start': 1},
                                         {'end': 4, 'entity': 'X', 'id': 'E2', 'location': 'B', 'start': 2},
                                         {'end': 4, 'entity': 'Y', 'id': 'E3', 'location': 'B', 'start': 2}]},
 'hark_household_policy_response': {'households': [{'assets': 0, 'baseline_consumption': 800, 'income': 1000},
                                                    {'assets': 5000, 'baseline_consumption': 1500, 'income': 2000}],
                                    'marginal_propensity_to_consume': 0.6,
                                    'transfer': 100},
 'igraph_link_analysis': {'directed': True,
                          'edges': [['A', 'B'], ['B', 'C'], ['C', 'A'], ['C', 'D']],
                          'nodes': ['A', 'B', 'C', 'D']},
 'issue_tree_coverage': {'branches': [{'evidence_count': 2, 'name': 'Revenue', 'weight': 0.5},
                                      {'evidence_count': 0, 'name': 'Cost', 'weight': 0.5}],
                         'root': 'Profit decline'},
 'negmas_bilateral_bargaining': {'buyer_ceiling': 75,
                                 'offers': [40, 50, 60, 70, 80],
                                 'seller_floor': 45,
                                 'seller_power': 0.5},
 'net_assessment_balance': {'dimensions': [{'competitor': 7, 'name': 'capital', 'own': 8, 'weight': 1},
                                           {'competitor': 9, 'name': 'distribution', 'own': 5, 'weight': 2}]},
 'open_spiel_policy_evaluation': {'column_policy': [0.4, 0.3, 0.3],
                                  'game_id': 'matrix_rps',
                                  'row_policy': [0.2, 0.5, 0.3]},
 'owlready2_ontology_summary': {'classes': ['Firm', 'Supplier', 'Customer'],
                                'subclass_relations': [['Supplier', 'Firm']]},
 'policyengine_transfer_counterfactual': {'baseline_transfer': 100,
                                          'income_threshold': 25000,
                                          'records': [{'income': 10000, 'weight': 2},
                                                      {'income': 50000, 'weight': 1},
                                                      {'income': 20000, 'weight': 3}],
                                          'reform_transfer': 150},
 'problog_evidence_probability': {'facts': [{'name': 'credible_source', 'probability': 0.8},
                                            {'name': 'corroborated', 'probability': 0.7}]},
 'pyagrum_bayesian_evidence': {'evidence': [{'name': 'E1', 'p_if_false': 0.2, 'p_if_true': 0.8},
                                            {'name': 'E2', 'p_if_false': 0.4, 'p_if_true': 0.6}],
                               'prior': 0.3},
 'pyblp_price_counterfactual': {'costs': [2, 3, 4],
                                'counterfactual_prices': [5.5, 6, 6.5],
                                'market_size': 1000,
                                'price_sensitivity': 0.5,
                                'prices': [5, 6, 7],
                                'qualities': [1, 1.4, 2],
                                'quality_sensitivity': 1},
 'pygambit_pure_equilibria': {'column_payoffs': [[3, 5], [0, 1]], 'row_payoffs': [[3, 0], [5, 1]]},
 'pymc_marketing_budget_allocation': {'budget': 20,
                                      'channels': [{'half_saturation': 10, 'name': 'search', 'scale': 100},
                                                   {'half_saturation': 5, 'name': 'social', 'scale': 70}],
                                      'step': 1},
 'pyshacl_graph_validation': {'data_turtle': '@prefix ex: <urn:ex:> . ex:a ex:p "v" .',
                              'shapes_turtle': '@prefix sh: <http://www.w3.org/ns/shacl#> . @prefix ex: <urn:ex:> . '
                                               'ex:S a sh:NodeShape ; sh:targetNode ex:a ; sh:property [ sh:path '
                                               'ex:p ; sh:minCount 1 ] .'},
 'rapidfuzz_record_collision': {'left': ['Alpha Co', 'Beta Ltd'],
                                'right': ['Alpha Company', 'Gamma'],
                                'threshold': 70},
 'rdflib_claim_evidence_graph': {'triples': [{'object': 'source1',
                                             'predicate': 'supported_by',
                                             'subject': 'claim1'},
                                            {'object': 'agency',
                                             'predicate': 'published_by',
                                             'subject': 'source1'}]},
 'red_team_challenge_matrix': {'assumptions': [{'impact': 0.9,
                                                'name': 'demand grows',
                                                'reversibility': 0.2,
                                                'uncertainty': 0.7},
                                               {'impact': 0.5,
                                                'name': 'cost stable',
                                                'reversibility': 0.8,
                                                'uncertainty': 0.4}]},
 'scikit_criteria_method_agreement': {'alternatives': ['A', 'B', 'C'],
                                      'matrix': [[8, 4], [6, 2], [9, 8]],
                                      'objectives': ['max', 'min'],
                                      'weights': [0.6, 0.4]},
 'scml_supply_chain_competition': {'demand': 100,
                                   'sale_price': 8,
                                   'suppliers': [{'capacity': 60,
                                                  'name': 'A',
                                                  'reliability': 0.9,
                                                  'unit_cost': 4},
                                                 {'capacity': 70,
                                                  'name': 'B',
                                                  'reliability': 0.98,
                                                  'unit_cost': 5}]},
 'source_reliability_matrix': {'sources': [{'access': 0.8,
                                            'corroboration': 0.9,
                                            'name': 'official',
                                            'recency': 0.8,
                                            'reliability': 0.9},
                                           {'access': 0.3,
                                            'corroboration': 0.1,
                                            'name': 'rumor',
                                            'recency': 1,
                                            'reliability': 0.2}]},
 'splink_entity_resolution': {'fields': ['name', 'city'],
                              'records': [{'city': 'Fuzhou', 'name': 'Alpha Co'},
                                          {'city': 'Fuzhou', 'name': 'Alpha Company'},
                                          {'city': 'Xiamen', 'name': 'Beta'}],
                              'threshold': 0.8},
 'taxcalc_policy_counterfactual': {'baseline_policy': [{'rate': 0.1, 'threshold': 0},
                                                       {'rate': 0.2, 'threshold': 50000}],
                                   'incomes': [10000, 30000, 80000],
                                   'reform_policy': [{'rate': 0.08, 'threshold': 0},
                                                     {'rate': 0.22, 'threshold': 50000}]},
 'value_driver_tree': {'base_value': 100,
                       'drivers': [{'change': 5, 'multiplier': 2, 'name': 'price'},
                                   {'change': -3, 'multiplier': 1, 'name': 'cost'}]},
 'z3_constraint_counterexample': {'bounds': {'x': {'maximum': 10, 'minimum': 0},
                                              'y': {'maximum': 10, 'minimum': 0}},
                                  'constraints': [{'coefficients': {'x': 1, 'y': 1},
                                                   'relation': '>=',
                                                   'rhs': 5},
                                                  {'coefficients': {'x': 1}, 'relation': '<=', 'rhs': 4}],
                                  'variables': ['x', 'y']}}


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(HANDLERS), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if set(FIXTURES) != set(HANDLERS):
        raise AssertionError(
            f"fixture mismatch missing={sorted(set(HANDLERS)-set(FIXTURES))} "
            f"extra={sorted(set(FIXTURES)-set(HANDLERS))}"
        )
    result = HANDLERS[args.mode](FIXTURES[args.mode])
    if result.get("mode") != args.mode or not finite_tree(result):
        raise AssertionError("result contract or finite-value contract failed")
    payload = {
        "status": "PASS",
        "mode": args.mode,
        "network_used": False,
        "arbitrary_code_used": False,
        "result": result,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
