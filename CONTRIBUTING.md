# Contributing to Real-Time Object Detection & Tracking

Thank you for your interest in contributing to this project! We welcome contributions from the community and appreciate your help in making this project better.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of computer vision and object detection

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/real-time-object-detection-tracking.git
   cd real-time-object-detection-tracking
   ```

2. **Create a development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📋 Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the GitHub issue template
- Provide detailed reproduction steps
- Include system information and error logs
- Attach relevant screenshots or videos

### ✨ Feature Requests
- Use the GitHub issue template
- Describe the feature and its benefits
- Provide use cases and examples
- Consider implementation complexity

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation updates
- Test coverage improvements

### 📚 Documentation
- README improvements
- Code comments and docstrings
- Tutorial creation
- API documentation

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

**Branch naming conventions:**
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add appropriate comments and docstrings
- Include type hints where applicable
- Write tests for new functionality

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_detector.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run linting
flake8 src/
black --check src/

# Run type checking
mypy src/
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add support for custom model loading

- Add ModelLoader class for dynamic model loading
- Support for custom YOLO model paths
- Add model validation and error handling
- Update documentation with examples"
```

**Commit message format:**
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 5. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## 📝 Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def detect_objects(self, frame: np.ndarray) -> List[Dict]:
    """Detect objects in a frame.
    
    Args:
        frame: Input frame as numpy array
        
    Returns:
        List of detection dictionaries
    """
    pass

# Use descriptive variable names
detection_results = self.model.predict(frame)
confidence_threshold = 0.5

# Use constants for magic numbers
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_DETECTIONS = 1000
```

### Documentation Style

```python
class YOLODetector:
    """YOLOv8-based object detector with optimization features.
    
    This class provides a high-performance interface for object detection
    using YOLOv8 models with various optimization techniques.
    
    Attributes:
        model: YOLO model instance
        device: Device to run inference on
        confidence_threshold: Minimum confidence for detections
        
    Example:
        >>> detector = YOLODetector(device="cuda")
        >>> detections = detector.detect(frame)
    """
```

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
import numpy as np
from src.detector import YOLODetector

class TestYOLODetector:
    """Test cases for YOLODetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization with default parameters."""
        detector = YOLODetector()
        assert detector.device == "cpu"
        assert detector.model is not None
    
    def test_detection_with_dummy_frame(self):
        """Test detection with a dummy frame."""
        detector = YOLODetector()
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(dummy_frame)
        
        assert isinstance(detections, list)
        # Add more specific assertions based on expected behavior
```

### Test Coverage

- Aim for >80% code coverage
- Test both success and failure cases
- Include edge cases and boundary conditions
- Mock external dependencies when appropriate

## 📊 Performance Considerations

### Benchmarking

When making performance-related changes:

```bash
# Run benchmarks before and after changes
python examples/benchmark.py --frames 100

# Compare different configurations
python examples/benchmark.py --compare-models
```

### Performance Guidelines

- Profile code before optimizing
- Use appropriate data structures
- Minimize memory allocations
- Consider GPU vs CPU trade-offs
- Document performance implications

## 🔍 Code Review Process

### For Contributors

1. **Self-review your code**
   - Check for typos and formatting issues
   - Ensure all tests pass
   - Verify documentation is up to date
   - Test your changes thoroughly

2. **Respond to feedback**
   - Address all review comments
   - Ask questions if something is unclear
   - Be open to suggestions and improvements

### For Reviewers

1. **Be constructive and respectful**
   - Provide specific, actionable feedback
   - Explain the reasoning behind suggestions
   - Acknowledge good practices

2. **Check for**
   - Code correctness and logic
   - Performance implications
   - Security considerations
   - Documentation completeness
   - Test coverage

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template and include:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10, Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Package versions: [e.g. torch 1.12.0, opencv 4.6.0]

**Additional context**
Any other context about the problem.
```

### Feature Requests

Use the feature request template and include:

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Include type hints
- Provide usage examples
- Document parameters and return values

### README Updates

When adding new features:
- Update the README with usage examples
- Add new command-line options
- Update the feature list
- Include performance benchmarks if applicable

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are updated
- [ ] Release notes are prepared

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different opinions and approaches

### Getting Help

- Check existing issues and discussions
- Use GitHub Discussions for questions
- Join our community chat (if available)
- Be patient with responses

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/real-time-object-detection-tracking/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/real-time-object-detection-tracking/discussions)
- **Email**: your.email@example.com

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributors page

Thank you for contributing to this project! 🎉
