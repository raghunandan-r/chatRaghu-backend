[1mdiff --git a/.gitignore b/.gitignore[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/.pre-commit-config.yaml b/.pre-commit-config.yaml[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/Dockerfile b/Dockerfile[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/Procfile b/Procfile[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/README.md b/README.md[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/app.py b/app.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/chatRaghu.code-workspace b/chatRaghu.code-workspace[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/docker-compose.yml b/docker-compose.yml[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/.gitignore b/evals_service/.gitignore[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/Dockerfile b/evals_service/Dockerfile[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/ISSUE_LOG.md b/evals_service/ISSUE_LOG.md[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/README.md b/evals_service/README.md[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/__init__.py b/evals_service/__init__.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/app.py b/evals_service/app.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/config.py b/evals_service/config.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators.py b/evals_service/evaluators.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/__init__.py b/evals_service/evaluators/__init__.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/base.py b/evals_service/evaluators/base.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/generate_with_context.py b/evals_service/evaluators/generate_with_context.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/generate_with_persona.py b/evals_service/evaluators/generate_with_persona.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/judgements.py b/evals_service/evaluators/judgements.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/models.py b/evals_service/evaluators/models.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/evaluators/relevance_check.py b/evals_service/evaluators/relevance_check.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/models.py b/evals_service/models.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/pyproject.toml b/evals_service/pyproject.toml[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/queue_manager.py b/evals_service/queue_manager.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/requirements.txt b/evals_service/requirements.txt[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mindex 45da12c..367ecc0[m
[1m--- a/evals_service/requirements.txt[m
[1m+++ b/evals_service/requirements.txt[m
[36m@@ -1,14 +1,15 @@[m
[31m-fastapi==0.104.1[m
[31m-uvicorn[standard]==0.24.0[m
[32m+[m[32mfastapi==0.110.0[m
[32m+[m[32muvicorn[standard]==0.29.0[m
 pydantic==2.8.2[m
[31m-pydantic-settings>=2.3.0[m
[31m-python-dotenv==1.0.0[m
[32m+[m[32mpydantic-settings==2.3.0[m
[32m+[m[32mpython-dotenv==1.0.1[m
 openai==1.70.0[m
[31m-pandas==2.1.4[m
[31m-pyarrow==14.0.2[m
[32m+[m[32mpandas==2.2.2[m
[32m+[m[32mpyarrow==16.1.0[m
 backoff==2.2.1[m
[31m-python-multipart==0.0.6[m
[31m-httpx>=0.25.2[m
[32m+[m[32mpython-multipart==0.0.7[m
[32m+[m[32mhttpx==0.27.0[m
[32m+[m[32mhttpcore==0.18.0[m
 structlog==23.1.0[m
 instructor==1.9.0[m
 [m
[36m@@ -22,5 +23,5 @@[m [mpytest-httpx==0.29.0[m
 [m
 # logging[m
 sentry-sdk==2.23.1[m
[31m-opik>=1.7.40[m
[32m+[m[32mopik==1.7.40[m
 python-json-logger==2.0.7[m
[1mdiff --git a/evals_service/run_evals.py b/evals_service/run_evals.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/setup.py b/evals_service/setup.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/storage.py b/evals_service/storage.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/utils/__init__.py b/evals_service/utils/__init__.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evals_service/utils/logger.py b/evals_service/utils/logger.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evaluation_client.py b/evaluation_client.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evaluation_models.py b/evaluation_models.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/evaluation_queue_manager.py b/evaluation_queue_manager.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/exit b/exit[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/get-pip.py b/get-pip.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/README.md b/graph/README.md[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/__init__.py b/graph/__init__.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/graph.py.backup b/graph/graph.py.backup[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/infrastructure.py b/graph/infrastructure.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/models.py b/graph/models.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/nodes.py b/graph/nodes.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mindex 6c4b828..1296612[m
[1m--- a/graph/nodes.py[m
[1m+++ b/graph/nodes.py[m
[36m@@ -911,8 +911,7 @@[m [mstreaming_graph = StreamingStateGraph([m
         "generate_with_persona": GenerateWithPersonaNode(name="generate_with_persona"),[m
     },[m
     edges={[m
[31m-        "relevance_check": {[m
[31m-            "CONTEXTUAL": "query_or_respond",[m
[32m+[m[32m        "relevance_check": {[m[41m            [m
             "IRRELEVANT": "few_shot_selector",[m
             "RELEVANT": "query_or_respond",[m
         },[m
[1mdiff --git a/graph/prompt_templates.json b/graph/prompt_templates.json[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/graph/retrieval.py b/graph/retrieval.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/pyproject.toml b/pyproject.toml[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/pytest.ini b/pytest.ini[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/railway.json b/railway.json[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/requirements.txt b/requirements.txt[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mindex c22e835..d1cf67b[m
[1m--- a/requirements.txt[m
[1m+++ b/requirements.txt[m
[36m@@ -1,47 +1,47 @@[m
[31m-openai>=1.60.2[m
[32m+[m[32mopenai==1.70.0[m
 pinecone-client==3.0.0[m
 [m
[31m-# Dependencies that can be flexible[m
[31m-distro>=1.8.0[m
[31m-tqdm>=4.65.0[m
[31m-packaging>=23.0[m
[31m-requests>=2.25.1[m
[31m-urllib3>=1.26.0[m
[31m-sqlalchemy>=1.4.0[m
[31m-aiohttp>=3.7.4[m
[31m-python-socketio>=5.0.0[m
[31m-jiter>=0.4.0,<1[m
[32m+[m[32m# Dependencies that can be flexible (now pinned)[m
[32m+[m[32mdistro==1.8.0[m
[32m+[m[32mtqdm==4.66.4[m
[32m+[m[32mpackaging==24.0[m
[32m+[m[32mrequests==2.32.3[m
[32m+[m[32murllib3==1.26.18[m
[32m+[m[32msqlalchemy==2.0.30[m
[32m+[m[32maiohttp==3.9.5[m
[32m+[m[32mpython-socketio==5.11.2[m
[32m+[m[32mjiter==0.4.0[m
 [m
 # Web framework core (keep strict versions)[m
[31m-fastapi==0.109.0[m
[31m-uvicorn[standard]==0.27.0[m
[31m-pydantic==2.7.4[m
[32m+[m[32mfastapi==0.110.0[m
[32m+[m[32muvicorn[standard]==0.29.0[m
[32m+[m[32mpydantic==2.8.2[m
 pydantic-core==2.18.4[m
 [m
[31m-# Web dependencies (can be flexible)[m
[31m-click>=8.1.7[m
[31m-h11>=0.14.0[m
[31m-httptools>=0.6.1[m
[31m-python-multipart>=0.0.7[m
[31m-websockets>=11.0.3[m
[31m-httpx>=0.24.1[m
[31m-httpcore>=0.17.0[m
[31m-certifi>=2023.7.22[m
[31m-importlib-metadata>=6.0.0[m
[31m-charset-normalizer>=2.0.0[m
[31m-starlette>=0.35.0[m
[32m+[m[32m# Web dependencies (now pinned)[m
[32m+[m[32mclick==8.1.7[m
[32m+[m[32mh11==0.14.0[m
[32m+[m[32mhttptools==0.6.1[m
[32m+[m[32mpython-multipart==0.0.7[m
[32m+[m[32mwebsockets==12.0[m
[32m+[m[32mhttpx==0.27.0[m
[32m+[m[32mhttpcore==0.18.0[m
[32m+[m[32mcertifi==2024.6.2[m
[32m+[m[32mimportlib-metadata==7.0.0[m
[32m+[m[32mcharset-normalizer==3.3.2[m
[32m+[m[32mstarlette==0.36.3[m
 [m
[31m-# Core dependencies (can be flexible)[m
[31m-python-dotenv>=1.0.0[m
[31m-numpy>=1.24.3[m
[31m-pandas>=2.2.0[m
[31m-backoff>=2.2.1[m
[31m-typing-extensions>=4.9.0[m
[31m-annotated-types>=0.5.0[m
[31m-anyio>=4.2.0[m
[31m-idna>=3.6[m
[31m-sniffio>=1.3.0[m
[31m-exceptiongroup>=1.0.0[m
[32m+[m[32m# Core dependencies (now pinned)[m
[32m+[m[32mpython-dotenv==1.0.1[m
[32m+[m[32mnumpy==1.26.4[m
[32m+[m[32mpandas==2.2.2[m
[32m+[m[32mbackoff==2.2.1[m
[32m+[m[32mtyping-extensions==4.11.0[m
[32m+[m[32mannotated-types==0.6.0[m
[32m+[m[32manyio==4.4.0[m
[32m+[m[32midna==3.7[m
[32m+[m[32msniffio==1.3.1[m
[32m+[m[32mexceptiongroup==1.2.1[m
 [m
 # Testing and Development tools (keep strict versions)[m
 pytest==8.2.2[m
[36m@@ -59,4 +59,4 @@[m [mgunicorn==21.2.0[m
 [m
 # logging[m
 sentry-sdk==2.23.1[m
[31m-opik==1.5.8[m
[32m+[m[32mopik==1.7.40[m
[1mdiff --git a/runtime.txt b/runtime.txt[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/tests/README.md b/tests/README.md[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/tests/__init__.py b/tests/__init__.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/tests/conftest.py b/tests/conftest.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/tests/test_graph.py b/tests/test_graph.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/tests/test_integration.py b/tests/test_integration.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/tests/test_unit.py b/tests/test_unit.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/utils/__init__.py b/utils/__init__.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mdiff --git a/utils/logger.py b/utils/logger.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
