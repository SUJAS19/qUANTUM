# Contributing to qUANTUM

Thank you for your interest in contributing to qUANTUM! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets

### Suggesting Features

Feature suggestions are welcome! Please:
- Check if the feature already exists
- Provide clear use cases
- Explain the expected behavior
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Write tests** (if applicable)
5. **Update documentation**
6. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of changes"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add type hints where appropriate

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/qUANTUM.git
cd qUANTUM

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/
```

## Project Structure

```
qUANTUM/
├── data/              # Data pipeline
├── strategies/        # Trading strategies
├── backtesting/       # Backtesting engine
├── screening/         # Stock screening
├── options/           # Options analysis
├── ml/               # ML models
├── utils/            # Utilities
├── tests/            # Unit tests
└── examples/         # Example scripts
```

## Areas for Contribution

- **New Strategies**: Implement new trading strategies
- **Data Sources**: Add support for new data providers
- **ML Models**: Add new forecasting models
- **Indicators**: Implement additional technical indicators
- **Optimization**: Improve performance and efficiency
- **Documentation**: Improve docs and examples
- **Tests**: Add unit tests and integration tests

## Code Review Process

1. Maintainers review PRs within 1-2 weeks
2. Feedback is provided for improvements
3. Once approved, PR is merged
4. Contributors are credited in CHANGELOG

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on improving the project

## Questions?

Feel free to open an issue for questions or discussions!

Thank you for contributing to qUANTUM! 🚀

