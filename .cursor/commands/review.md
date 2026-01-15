# Python Code Review Guide

## Overview
This guide provides a comprehensive checklist for reviewing Python code in this project. Follow these guidelines to ensure code quality, consistency, and adherence to project standards.

---

## 1. Code Style & Formatting

### Basic Style
- [ ] Code follows Black formatting (88 character line limit)
- [ ] Imports are sorted with isort
- [ ] PEP 8 naming conventions are followed:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- [ ] File/directory names use lowercase with underscores (e.g., `user_routes.py`)
- [ ] Descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`)

### Functional Programming
- [ ] Functional, declarative programming is preferred over classes where possible
- [ ] Code is modular and avoids duplication
- [ ] RORO pattern (Receive an Object, Return an Object) is used where appropriate

---

## 2. Type Hints & Documentation

### Type Hints
- [ ] All function parameters have type hints
- [ ] All function returns have type hints
- [ ] `Optional[Type]` is used instead of `Type | None`
- [ ] Types are imported from `typing` module
- [ ] Custom types are defined in `types.py`
- [ ] Generic types use `TypeVar` appropriately
- [ ] Duck typing uses `Protocol` when needed

### Documentation
- [ ] All functions have Google-style docstrings
- [ ] Public APIs are documented
- [ ] Complex logic has inline comments
- [ ] Docstrings include:
  - Brief description
  - Args with types
  - Returns with type
  - Raises with exception types
  - Examples (where helpful)

**Example:**
```python
def process_user_data(user_id: int, options: Optional[dict] = None) -> dict:
    """Process user data with optional configuration.
    
    Args:
        user_id: The unique identifier for the user
        options: Optional configuration dictionary with processing parameters
        
    Returns:
        A dictionary containing processed user data with keys: 'status', 'data', 'timestamp'
        
    Raises:
        ValueError: If user_id is invalid or user not found
        ProcessingError: If data processing fails
        
    Example:
        >>> result = process_user_data(123, {"validate": True})
        >>> print(result['status'])
        'success'
    """
    # Implementation here
```

---

## 3. Error Handling & Edge Cases

### Error Handling Priority
- [ ] Errors and edge cases are handled at the beginning of functions
- [ ] Early returns are used for error conditions (avoid deeply nested if statements)
- [ ] Happy path is placed last in the function
- [ ] Guard clauses handle preconditions and invalid states early
- [ ] No unnecessary `else` statements (use if-return pattern)

### Error Implementation
- [ ] Custom exception classes are used for domain-specific errors
- [ ] Proper try-except blocks are implemented
- [ ] Error logging is comprehensive and informative
- [ ] User-friendly error messages are returned
- [ ] All edge cases are properly handled
- [ ] Custom error types or error factories ensure consistent error handling

**Good Example:**
```python
async def get_user_profile(user_id: int) -> dict:
    """Retrieve user profile with proper error handling."""
    # Handle edge cases first
    if user_id <= 0:
        logger.error(f"Invalid user_id: {user_id}")
        raise ValueError("User ID must be positive")
    
    # Check preconditions
    if not await user_exists(user_id):
        logger.warning(f"User not found: {user_id}")
        raise UserNotFoundError(f"User {user_id} does not exist")
    
    # Happy path last
    try:
        profile = await fetch_user_profile(user_id)
        return profile
    except DatabaseError as e:
        logger.error(f"Database error fetching user {user_id}: {e}")
        raise DataFetchError("Failed to retrieve user profile") from e
```

**Bad Example:**
```python
async def get_user_profile(user_id: int) -> dict:
    """Retrieve user profile with poor error handling."""
    try:
        if user_id > 0:
            if await user_exists(user_id):
                profile = await fetch_user_profile(user_id)
                return profile
            else:
                raise UserNotFoundError("User not found")
        else:
            raise ValueError("Invalid user ID")
    except Exception as e:
        logger.error(str(e))
        raise
```

---

## 4. FastAPI Specific

### Route Structure
- [ ] Routes are organized by domain in separate modules
- [ ] Proper HTTP methods are used (GET, POST, PUT, DELETE, PATCH)
- [ ] Correct HTTP status codes are returned
- [ ] Request/response models use Pydantic
- [ ] Input validation is implemented
- [ ] API endpoints are documented with proper descriptions

### Dependencies & Injection
- [ ] FastAPI's dependency injection is used for shared resources
- [ ] Dependencies are properly typed
- [ ] Database sessions use dependency injection
- [ ] Authentication uses dependency injection
- [ ] No global state is used

### Async Operations
- [ ] `async def` is used for I/O-bound operations
- [ ] `def` is used for CPU-bound operations
- [ ] No blocking operations in async routes
- [ ] Proper async/await usage throughout
- [ ] Background tasks are used for long-running operations

### Middleware
- [ ] Logging middleware is implemented
- [ ] Error monitoring is in place
- [ ] Performance optimization middleware is used
- [ ] CORS is properly configured
- [ ] Security headers are set

### Performance
- [ ] Asynchronous operations are used where appropriate
- [ ] Caching is implemented for expensive operations
- [ ] Database queries are optimized
- [ ] Proper connection pooling is used
- [ ] Response times are monitored

**Example Route:**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db
from ..models import UserResponse, UserCreate
from ..auth import get_current_user

router = APIRouter(prefix="/users", tags=["users"])

@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    description="Create a new user account"
)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
) -> UserResponse:
    """Create a new user with validation and proper error handling."""
    # Validation and error handling first
    if not user_data.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required"
        )
    
    if await email_exists(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Happy path
    try:
        new_user = await create_user_in_db(db, user_data)
        return UserResponse.from_orm(new_user)
    except DatabaseError as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
```

---

## 5. Database & ORM

### SQLAlchemy Usage
- [ ] Proper ORM models are defined
- [ ] Relationships are correctly implemented
- [ ] Indexes are used appropriately
- [ ] Connection pooling is configured
- [ ] Migrations are implemented with Alembic
- [ ] Transactions are handled properly

### Query Optimization
- [ ] N+1 queries are avoided (use eager loading)
- [ ] Queries are optimized for performance
- [ ] Proper pagination is implemented
- [ ] Database errors are handled gracefully
- [ ] Connection leaks are prevented

---

## 6. Security

### Input Validation
- [ ] All user inputs are validated
- [ ] Pydantic models are used for validation
- [ ] SQL injection is prevented (use ORM/parameterized queries)
- [ ] XSS attacks are prevented
- [ ] CSRF protection is implemented

### Authentication & Authorization
- [ ] JWT authentication is properly implemented
- [ ] Passwords are hashed (bcrypt/argon2)
- [ ] Role-based access control is used
- [ ] Session management is secure
- [ ] OAuth2 is correctly implemented (if used)

### General Security
- [ ] CORS is properly configured
- [ ] Rate limiting is implemented
- [ ] Security headers are set
- [ ] Secrets are not hardcoded (use environment variables)
- [ ] HTTPS is enforced in production
- [ ] Logging doesn't expose sensitive data

---

## 7. Testing

### Test Coverage
- [ ] Unit tests are written for all functions
- [ ] Integration tests cover API endpoints
- [ ] Edge cases are tested
- [ ] Error scenarios are tested
- [ ] Test coverage is above 80%

### Test Quality
- [ ] Tests use pytest
- [ ] Proper fixtures are implemented
- [ ] Mocking is used appropriately (pytest-mock)
- [ ] Tests are independent and can run in any order
- [ ] Test names are descriptive
- [ ] Async tests use `@pytest.mark.asyncio`

**Example Test:**
```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_user_success(async_client: AsyncClient, mock_db):
    """Test successful user creation."""
    user_data = {
        "email": "test@example.com",
        "name": "Test User"
    }
    
    response = await async_client.post("/users/", json=user_data)
    
    assert response.status_code == 201
    assert response.json()["email"] == user_data["email"]
    assert "id" in response.json()

@pytest.mark.asyncio
async def test_create_user_duplicate_email(async_client: AsyncClient, existing_user):
    """Test user creation with duplicate email."""
    user_data = {
        "email": existing_user.email,
        "name": "Another User"
    }
    
    response = await async_client.post("/users/", json=user_data)
    
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]
```

---

## 8. Performance & Optimization

### Code Efficiency
- [ ] Expensive operations are cached
- [ ] Background tasks are used for heavy operations
- [ ] Database queries are optimized
- [ ] Proper pagination is implemented
- [ ] Memory usage is considered
- [ ] No unnecessary computations in loops

### Monitoring
- [ ] Logging is comprehensive but not excessive
- [ ] Performance metrics are tracked
- [ ] Error rates are monitored
- [ ] Response times are logged
- [ ] Resource usage is tracked

---

## 9. Project Structure & Organization

### File Organization
- [ ] Code follows the project structure (src-layout if applicable)
- [ ] Routes are organized by domain
- [ ] Models are in separate modules
- [ ] Utilities are properly organized
- [ ] Configuration is in proper location
- [ ] Tests mirror source structure

### Dependencies
- [ ] New dependencies are justified
- [ ] Dependency versions are pinned
- [ ] Dependencies are added to requirements.txt
- [ ] No unused imports
- [ ] Absolute imports are used over relative imports

---

## 10. Code Quality Checks

### Before Approval
- [ ] Code runs without errors
- [ ] All tests pass
- [ ] Linting passes (flake8, pylint, mypy)
- [ ] Type checking passes (mypy)
- [ ] Code coverage meets threshold
- [ ] No security vulnerabilities introduced
- [ ] Performance hasn't regressed
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### Common Issues to Watch For
- [ ] No print statements (use logging)
- [ ] No commented-out code
- [ ] No TODO comments without tickets
- [ ] No debug code left in
- [ ] No hardcoded values (use config/env vars)
- [ ] No overly complex functions (consider breaking down)
- [ ] No global variables
- [ ] No mutable default arguments

---

## 11. Review Process

### As a Reviewer
1. **First Pass**: Read through the entire PR to understand the changes
2. **Architecture Review**: Check if the approach makes sense
3. **Detailed Review**: Go through each file checking against this guide
4. **Test Review**: Verify test coverage and quality
5. **Documentation**: Ensure docs are updated
6. **Feedback**: Provide constructive, specific feedback with examples

### Feedback Guidelines
- Be respectful and constructive
- Explain the "why" behind suggestions
- Provide code examples for improvements
- Distinguish between "must fix" and "nice to have"
- Acknowledge good practices in the code
- Ask questions rather than making demands

**Good Feedback:**
> "Consider using early returns here to improve readability. For example:
> ```python
> if not user_id:
>     raise ValueError("User ID required")
> ```
> This avoids the nested if statement on line 45."

**Bad Feedback:**
> "This is wrong, fix it."

---

## 12. Quick Reference Checklist

### Essential Checks ⚡
- [ ] Type hints on all functions
- [ ] Error handling at function start
- [ ] Early returns for error conditions
- [ ] Proper async/await usage
- [ ] Input validation with Pydantic
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No security issues
- [ ] Follows project structure

### Code Smells 🚨
- ❌ Deeply nested if statements
- ❌ Functions longer than 50 lines
- ❌ Commented-out code
- ❌ print() statements
- ❌ Hardcoded secrets or URLs
- ❌ Global variables
- ❌ Mutable default arguments
- ❌ Bare except clauses
- ❌ Missing type hints
- ❌ No docstrings on public functions

---

## Resources

- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [SQLAlchemy Best Practices](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- Project Rules: `.cursor/rules/` directory

---

## Summary

A good code review ensures:
- ✅ Code is readable and maintainable
- ✅ Errors are handled properly
- ✅ Tests provide adequate coverage
- ✅ Security best practices are followed
- ✅ Performance is acceptable
- ✅ Documentation is clear
- ✅ Project standards are met

Remember: The goal of code review is to improve code quality and share knowledge, not to criticize. Review with empathy and focus on creating better software together.
