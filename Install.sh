python -m pip install -e '.[torch]'
python -m compileall -q kubu_hai tests
python -m unittest discover -s tests -v
