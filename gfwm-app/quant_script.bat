@echo off
python -m data_science.quant.get_data
python -m data_science.quant.calculate_cov
python -m data_science.quant.plot_markowitz_bullet