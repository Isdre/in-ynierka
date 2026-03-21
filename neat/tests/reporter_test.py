import pytest
from unittest.mock import Mock
from neat.reporter import ReporterSet, BaseReporter

def test_add_and_remove_reporter():
    reporter_set = ReporterSet()
    mock_reporter = Mock(spec=BaseReporter)

    reporter_set.add(mock_reporter)
    assert mock_reporter in reporter_set.reporters

    reporter_set.remove(mock_reporter)
    assert mock_reporter not in reporter_set.reporters

def test_start_generation():
    reporter_set = ReporterSet()
    mock_reporter = Mock(spec=BaseReporter)
    reporter_set.add(mock_reporter)

    reporter_set.start_generation(1)
    mock_reporter.start_generation.assert_called_once_with(1)

def test_end_generation():
    reporter_set = ReporterSet()
    mock_reporter = Mock(spec=BaseReporter)
    reporter_set.add(mock_reporter)

    config = Mock()
    population = Mock()
    species = Mock()
    reporter_set.end_generation(config, population, species)
    mock_reporter.end_generation.assert_called_once_with(config, population, species)

def test_found_solution():
    reporter_set = ReporterSet()
    mock_reporter = Mock(spec=BaseReporter)
    reporter_set.add(mock_reporter)

    config = Mock()
    best_genome = Mock()
    reporter_set.found_solution(config, 10, best_genome)
    mock_reporter.found_solution.assert_called_once_with(config, 10, best_genome)

def test_info():
    reporter_set = ReporterSet()
    mock_reporter = Mock(spec=BaseReporter)
    reporter_set.add(mock_reporter)

    reporter_set.info("Test message")
    mock_reporter.info.assert_called_once_with("Test message")

def test_warning():
    reporter_set = ReporterSet()
    mock_reporter = Mock(spec=BaseReporter)
    reporter_set.add(mock_reporter)

    reporter_set.warning("Warning message")
    mock_reporter.warning.assert_called_once_with("Warning message")