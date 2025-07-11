"""
Configuration settings for the Cluster-Form API.
Defines different configuration classes for development, testing, and production.
"""


class Config:
    """Base configuration class with common settings."""

    TESTING = False


class DevelopmentConfig(Config):
    """Development configuration settings."""

    pass


class TestingConfig(Config):
    """Testing configuration settings."""

    TESTING = True


class ProductionConfig(Config):
    """Production configuration settings."""

    pass
