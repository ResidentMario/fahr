`fahr` is beta software, and is still under active development.

## Development

### Cloning

To work on `fahr` locally, you will need to clone it.

```sh
git clone https://github.com/ResidentMario/fahr.git
```

You can then set up your own branch version of the code, and work on your changes for a pull request from there.

```sh
cd fahr
git checkout -B new-branch-name
```

### Environment

I strongly recommend creating a new virtual environment when working on `fahr` (e.g. not using the base system Python). I recommend doing so with [`conda`](https://conda.io/) or [`pipenv`](https://github.com/pypa/pipenv).

You should then create an [editable install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs) of `fahr` suitable for tweaking and further development. Do this by running:

```sh
pip install -e fahr .[develop]
```

Note that `fahr` is currently Python 3.6+ only.

### Tests

The `tests` folder contains tests. You may run the core tests by running `pytest test_fahr.py` and the CLI tests by running `pytest test_cli.py`.

## Documentation

`fahr` documentation is generated via [`sphinx`](http://www.sphinx-doc.org/en/stable/index.html) and served using [GitHub Pages](https://pages.github.com/). You can access it [here](https://residentmario.github.io/fahr/index.html).

The website is automatically updated whenever the `gh-pages` branch on GitHub is updated. `gh-pages` is an orphan branch containing only documentation. The root documentation files are kept in the `master` branch, then pushed to `gh-pages` by doing the following:

```sh
git checkout gh-pages
rm -rf *
git checkout master -- docs/ fahr/
cd docs; make html; cd ..
mv ./docs/_build/html/* ./
rm -rf docs fahr
```

So to update the documentation, edit the `.rst` files in the `docs` folder, then run `make html` there from the command line (optionally also running `make clean` beforehand). Then follow the procedure linked above to make these changes live.