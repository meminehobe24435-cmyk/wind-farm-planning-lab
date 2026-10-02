# wind-farm-planning-lab —— 常用命令
# Windows 上 make 可能不可用，等价命令见 README「快速开始」。

PY ?= python
export PYTHONIOENCODING := utf-8

.PHONY: help test test-v demo cli-report clean

help:
	@echo "make test      - 运行全部单元测试"
	@echo "make test-v    - 运行全部单元测试（详细）"
	@echo "make demo      - 端到端示例，输出到 reports/"
	@echo "make clean     - 清理缓存与报告产物"

test:
	$(PY) -m pytest tests -q

test-v:
	$(PY) -m pytest tests -v

demo:
	$(PY) scripts/demo.py

clean:
	-rm -rf .pytest_cache
	-find . -name "__pycache__" -type d -exec rm -rf {} +
	-rm -f reports/*.png reports/*.csv reports/*.md
