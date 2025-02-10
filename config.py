class Config:
    TESTING = False


class DevelopmentConfig(Config):
    pass


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass
