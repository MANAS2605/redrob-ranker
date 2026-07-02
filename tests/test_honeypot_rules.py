#!/usr/bin/env python3
"""
test_honeypot_rules.py — Unit tests for honeypot detection rules.

Uses synthetic candidate profiles to verify each rule fires correctly
and doesn't false-positive on legitimate profiles.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from precompute.precompute_honeypot_flags import (
    check_yoe_mismatch,
    check_expert_skill_low_duration,
    check_impossible_edu_sequence,
    check_edu_career_anomaly,
    check_multiple_current_roles,
    check_career_date_overlap,
    check_title_description_mismatch,
    detect_honeypot,
)


def _make_candidate(**overrides) -> dict:
    """Create a minimal valid candidate with optional overrides."""
    base = {
        "candidate_id": "CAND_0000099",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Software Engineer",
            "summary": "Experienced software engineer with ML background.",
            "location": "Bangalore, Karnataka",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Software Engineer",
            "current_company": "TechCorp",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "TechCorp",
                "title": "Software Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 54,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Built backend services and data pipelines using Python and SQL.",
            },
            {
                "company": "StartupABC",
                "title": "Junior Developer",
                "start_date": "2018-06-01",
                "end_date": "2019-12-31",
                "duration_months": 18,
                "is_current": False,
                "industry": "Fintech",
                "company_size": "11-50",
                "description": "Developed web applications using React and Node.js.",
            },
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 25, "duration_months": 48},
            {"name": "SQL", "proficiency": "intermediate", "endorsements": 10, "duration_months": 30},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-15",
            "open_to_work_flag": True,
            "profile_views_received_30d": 15,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.65,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"Python": 78.5},
            "connection_count": 250,
            "endorsements_received": 30,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 15, "max": 25},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 45,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.85,
            "offer_acceptance_rate": 0.75,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if key in base:
            if isinstance(base[key], dict) and isinstance(value, dict):
                base[key].update(value)
            else:
                base[key] = value
        elif "." in key:
            parts = key.split(".")
            obj = base
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = value
    
    return base


# ===== Test YOE Mismatch =====

def test_yoe_mismatch_no_flag():
    """Normal candidate: YOE matches career history."""
    c = _make_candidate()
    # 54 + 18 = 72 months = 6.0 years, matches years_of_experience=6.0
    score, reason = check_yoe_mismatch(c)
    assert score == 0.0, f"Should not flag: {reason}"
    print("PASS test_yoe_mismatch_no_flag")


def test_yoe_mismatch_large_gap():
    """Honeypot: claims 15 years but career only sums to 6."""
    c = _make_candidate()
    c["profile"]["years_of_experience"] = 15.0
    score, reason = check_yoe_mismatch(c)
    assert score > 0, f"Should flag large YOE mismatch"
    assert "15.0y" in reason
    print("PASS test_yoe_mismatch_large_gap")


def test_yoe_mismatch_small_gap_ok():
    """Normal: 2-year gap is within tolerance (gaps, sabbaticals)."""
    c = _make_candidate()
    c["profile"]["years_of_experience"] = 8.0  # 6.0 from career + 2y gap
    score, reason = check_yoe_mismatch(c)
    assert score == 0.0, f"Should not flag small gap: {reason}"
    print("PASS test_yoe_mismatch_small_gap_ok")


# ===== Test Expert Skill Low Duration =====

def test_expert_low_duration_flag():
    """Honeypot: expert proficiency with only 1 month of use."""
    c = _make_candidate(skills=[
        {"name": "FAISS", "proficiency": "expert", "endorsements": 50, "duration_months": 1},
    ])
    score, reason = check_expert_skill_low_duration(c)
    assert score > 0, "Should flag expert with 1mo"
    assert "FAISS" in reason
    print("PASS test_expert_low_duration_flag")


def test_expert_missing_duration_no_flag():
    """Normal: expert proficiency but duration_months is missing (optional field)."""
    c = _make_candidate(skills=[
        {"name": "FAISS", "proficiency": "expert", "endorsements": 50},  # no duration_months
    ])
    score, reason = check_expert_skill_low_duration(c)
    assert score == 0.0, f"Should NOT flag when duration is missing: {reason}"
    print("PASS test_expert_missing_duration_no_flag")


def test_advanced_reasonable_duration_no_flag():
    """Normal: advanced proficiency with 24 months — perfectly fine."""
    c = _make_candidate(skills=[
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
    ])
    score, reason = check_expert_skill_low_duration(c)
    assert score == 0.0, f"Should not flag reasonable duration: {reason}"
    print("PASS test_advanced_reasonable_duration_no_flag")


# ===== Test Impossible Education Sequence =====

def test_impossible_edu_sequence_phd_before_btech():
    """Honeypot: Ph.D completed before B.Tech."""
    c = _make_candidate(education=[
        {"institution": "Uni A", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2018, "end_year": 2022, "tier": "tier_2"},
        {"institution": "Uni B", "degree": "Ph.D", "field_of_study": "ML",
         "start_year": 2010, "end_year": 2015, "tier": "tier_1"},
    ])
    score, reason = check_impossible_edu_sequence(c)
    assert score > 0, f"Should flag Ph.D before B.Tech: {reason}"
    print("PASS test_impossible_edu_sequence_phd_before_btech")


def test_normal_edu_sequence():
    """Normal: B.Tech then M.Tech then Ph.D in correct order."""
    c = _make_candidate(education=[
        {"institution": "Uni A", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2010, "end_year": 2014, "tier": "tier_2"},
        {"institution": "Uni B", "degree": "M.Tech", "field_of_study": "CS",
         "start_year": 2014, "end_year": 2016, "tier": "tier_1"},
    ])
    score, reason = check_impossible_edu_sequence(c)
    assert score == 0.0, f"Should not flag normal sequence: {reason}"
    print("PASS test_normal_edu_sequence")


# ===== Test Education-Career Anomaly =====

def test_ug_after_career_flag():
    """Honeypot: undergrad degree completing 6 years after career start."""
    c = _make_candidate(
        education=[
            {"institution": "Uni", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2016, "end_year": 2020, "tier": "tier_3"},
        ],
        career_history=[
            {"company": "Corp", "title": "Engineer", "start_date": "2014-01-01",
             "end_date": None, "duration_months": 72, "is_current": True,
             "industry": "Software", "company_size": "201-500",
             "description": "Built systems."},
        ],
    )
    score, reason = check_edu_career_anomaly(c)
    assert score > 0, f"Should flag UG after career start: {reason}"
    print("PASS test_ug_after_career_flag")


def test_postgrad_while_working_no_flag():
    """Normal: M.Tech while working — perfectly legitimate."""
    c = _make_candidate(
        education=[
            {"institution": "Uni", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2010, "end_year": 2014, "tier": "tier_2"},
            {"institution": "Uni2", "degree": "M.Tech", "field_of_study": "CS",
             "start_year": 2018, "end_year": 2020, "tier": "tier_1"},
        ],
        career_history=[
            {"company": "Corp", "title": "Engineer", "start_date": "2014-06-01",
             "end_date": None, "duration_months": 72, "is_current": True,
             "industry": "Software", "company_size": "201-500",
             "description": "Built systems."},
        ],
    )
    score, reason = check_edu_career_anomaly(c)
    assert score == 0.0, f"Should NOT flag postgrad while working: {reason}"
    print("PASS test_postgrad_while_working_no_flag")


# ===== Test Multiple Current Roles =====

def test_multiple_current_flag():
    """Suspicious: 3 roles marked as current."""
    c = _make_candidate(career_history=[
        {"company": "A", "title": "Dev", "start_date": "2023-01-01", "end_date": None,
         "duration_months": 18, "is_current": True, "industry": "Software",
         "company_size": "51-200", "description": "Code."},
        {"company": "B", "title": "Dev", "start_date": "2022-01-01", "end_date": None,
         "duration_months": 30, "is_current": True, "industry": "Software",
         "company_size": "51-200", "description": "Code."},
        {"company": "C", "title": "Dev", "start_date": "2021-01-01", "end_date": None,
         "duration_months": 42, "is_current": True, "industry": "Software",
         "company_size": "51-200", "description": "Code."},
    ])
    score, reason = check_multiple_current_roles(c)
    assert score > 0, "Should flag multiple current roles"
    print("PASS test_multiple_current_flag")


def test_single_current_no_flag():
    """Normal: exactly one current role."""
    c = _make_candidate()  # default has 1 current
    score, reason = check_multiple_current_roles(c)
    assert score == 0.0, f"Should not flag single current: {reason}"
    print("PASS test_single_current_no_flag")


# ===== Test Overall Detection =====

def test_clean_candidate_low_score():
    """A legitimate candidate should have near-zero implausibility."""
    c = _make_candidate()
    result = detect_honeypot(c)
    assert result["implausibility_score"] < 0.1, \
        f"Clean candidate scored {result['implausibility_score']}: {result['rule_reasons']}"
    print("PASS test_clean_candidate_low_score")


def test_stacked_honeypot_high_score():
    """A candidate with multiple red flags should score high."""
    c = _make_candidate(
        skills=[
            {"name": "FAISS", "proficiency": "expert", "endorsements": 99, "duration_months": 0},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 99, "duration_months": 1},
        ],
        education=[
            {"institution": "Uni", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2018, "end_year": 2022, "tier": "tier_2"},
            {"institution": "Uni2", "degree": "Ph.D", "field_of_study": "ML",
             "start_year": 2010, "end_year": 2015, "tier": "tier_1"},
        ],
    )
    c["profile"]["years_of_experience"] = 20.0  # but career sums to ~6y
    
    result = detect_honeypot(c)
    assert result["implausibility_score"] >= 0.5, \
        f"Multi-flag candidate only scored {result['implausibility_score']}"
    print("PASS test_stacked_honeypot_high_score")


# ===== Run all tests =====

if __name__ == "__main__":
    tests = [
        test_yoe_mismatch_no_flag,
        test_yoe_mismatch_large_gap,
        test_yoe_mismatch_small_gap_ok,
        test_expert_low_duration_flag,
        test_expert_missing_duration_no_flag,
        test_advanced_reasonable_duration_no_flag,
        test_impossible_edu_sequence_phd_before_btech,
        test_normal_edu_sequence,
        test_ug_after_career_flag,
        test_postgrad_while_working_no_flag,
        test_multiple_current_flag,
        test_single_current_no_flag,
        test_clean_candidate_low_score,
        test_stacked_honeypot_high_score,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"FAIL {test.__name__}: EXCEPTION: {e}")
            failed += 1
    
    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    
    if failed > 0:
        sys.exit(1)
