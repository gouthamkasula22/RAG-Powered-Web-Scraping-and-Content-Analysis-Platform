# Test Suite Organization

This document explains the organization of the test suite for the Web Content Analysis project.

## Test Directory Structure

```
tests/
├── __init__.py           # Package indicator
├── conftest.py           # Shared pytest fixtures and configuration
├── integration/          # Integration tests between components
├── performance/          # Performance and load testing
├── rag_integration/      # RAG-specific integration tests
├── security/             # Security testing
├── ui/                   # UI component tests
└── unit/                 # Unit tests for individual components
```

## Test Categories

### Unit Tests (`tests/unit/`)
Tests for individual components in isolation, typically testing a single class or function:
- Domain models testing
- Service method testing
- Utility function testing

### Integration Tests (`tests/integration/`)
Tests that verify the interaction between multiple components:
- API endpoint tests
- Database interaction tests
- Service integration tests

### RAG Integration Tests (`tests/rag_integration/`)
Tests specific to the Retrieval-Augmented Generation functionality:
- `test_rag_integration.py`: Basic RAG system functionality
- `test_rag_llm.py`: Integration between RAG and LLM providers
- `test_rag_features.py`: Specific RAG features testing
- `test_bulk_rag_integration.py`: Bulk analysis with RAG

### UI Tests (`tests/ui/`)
Tests focused on UI components:
- `test_trash_icon_ui.py`: UI demonstration for trash icon
- `test_website_deletion.py`: Website deletion UI functionality

### Performance Tests (`tests/performance/`)
Tests for measuring and validating performance characteristics:
- Load testing
- Response time testing
- Resource usage monitoring

### Security Tests (`tests/security/`)
Tests for security features and validations:
- Input validation
- Authentication/authorization
- Secure data handling

## Running Tests

To run all tests:
```
pytest
```

To run a specific category of tests:
```
pytest tests/unit/
pytest tests/integration/
pytest tests/rag_integration/
```

To run a specific test file:
```
pytest tests/rag_integration/test_rag_features.py
```

## Adding New Tests

When adding new tests:
1. Place them in the appropriate directory based on their category
2. Follow the naming convention: `test_*.py` for files and `test_*` for functions
3. Update this README if you create a new test category or change the structure

## Test Dependencies

All test files have been updated to properly include their dependencies and project root path for imports. Each test should be able to run independently or as part of the test suite.
