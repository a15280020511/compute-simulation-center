import importlib.util
import unittest
import numpy as np
from calibration_assurance import coverage_evaluation, conditional_coverage_check, parameter_identifiability_check, predictive_check
from cvxpy_assurance import constraint_residual_audit
from feedback_executor import evaluate
from state_estimation import kalman_filter, scalar_nonlinear_filter
from surrogate_experiment import latin_hypercube_design, surrogate_error_validation

class AccuracyEnhancementTests(unittest.TestCase):
    def test_coverage_and_conditional(self):
        r=coverage_evaluation([1,2,3],[0,1,2],[2,3,4],.9); self.assertEqual(r['empirical_coverage'],1.0)
        c=conditional_coverage_check([1]*40,[0]*40,[2]*40,['a']*20+['b']*20,.9); self.assertEqual(len(c['groups']),2)
    def test_identifiability_and_predictive(self):
        self.assertTrue(parameter_identifiability_check([[1,0],[0,1],[1,1]])['identifiable'])
        self.assertIn('observed_90_coverage',predictive_check([[1,2],[1.1,1.9],[.9,2.1]],[1,2]))
    def test_residual_and_feedback(self):
        self.assertTrue(constraint_residual_audit([1,2],[0,0],[2,3],3)['feasible_within_tolerance'])
        out=evaluate([{'record_type':'realized_outcome','prediction':.8,'realized':1,'kind':'probability'}]); self.assertAlmostEqual(out['brier'],.04)
    def test_state_and_design(self):
        out=kalman_filter([[1.0],[1.2]],[[1]],[[1]],[[.1]],[[.2]],[0],[[1]]); self.assertEqual(len(out['states']),2)
        self.assertEqual(len(scalar_nonlinear_filter([1,1.1],'particle_filter',.1,.2,1,1,particles=100)['states']),2)
        self.assertEqual(len(latin_hypercube_design([[0,1],[0,2]],10)['design']),10)
        self.assertTrue(surrogate_error_validation([1,2],[1,2])['publish_allowed'])
    @unittest.skipUnless(importlib.util.find_spec('cvxpy'), 'optional CVXPY not installed')
    def test_cvxpy_smoke(self):
        from cvxpy_assurance import convex_resource_allocation
        self.assertLessEqual(convex_resource_allocation([1,2],[0,0],[10,10],5)['maximum_constraint_violation'],1e-5)

if __name__=='__main__': unittest.main()
