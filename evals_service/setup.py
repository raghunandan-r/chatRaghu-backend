from setuptools import setup, find_packages

setup(
    name="evals_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Keep requirements.txt as source of truth
        # This just enables pip install -e .
    ],
    python_requires=">=3.11",
    author="Raghu",
    description="Evaluation service for ChatRaghu",
    package_data={
        "evals_service": [
            "secrets/*",
            "eval_results/*",
            "audit_data/*",
        ],
    },
    include_package_data=True,
)
