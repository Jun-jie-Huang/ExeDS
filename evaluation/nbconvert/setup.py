#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import io
import sys

from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.develop import develop

from io import BytesIO
from urllib.request import urlopen

from distutils.cmd import Command
from distutils.command.build import build
from distutils.command.sdist import sdist

# the name of the package
name = 'nbconvert'

pjoin = os.path.join
here = os.path.abspath(os.path.dirname(__file__))
pkg_root = pjoin(here, name)

packages = []
for d, _, _ in os.walk(pjoin(here, name)):
    if os.path.exists(pjoin(d, '__init__.py')):
        packages.append(d[len(here)+1:].replace(os.path.sep, '.'))

package_data = {
    'nbconvert.filters' : ['marked.js'],
    'nbconvert.resources' : ['style.min.css'],
    'nbconvert' : [
        'tests/files/*.*',
        'tests/exporter_entrypoint/*.py',
        'tests/exporter_entrypoint/*/*.*',
        'exporters/tests/files/*.*',
        'preprocessors/tests/files/*.*',
    ],
}

notebook_css_version = '5.4.0'
notebook_css_url = "https://cdn.jupyter.org/notebook/%s/style/style.min.css" % notebook_css_version

jupyterlab_css_version = '3.1.11'
jupyterlab_css_url = "https://unpkg.com/@jupyterlab/nbconvert-css@%s/style/index.css" % jupyterlab_css_version

jupyterlab_theme_light_version = '3.1.11'
jupyterlab_theme_light_url = "https://unpkg.com/@jupyterlab/theme-light-extension@%s/style/variables.css" % jupyterlab_theme_light_version

jupyterlab_theme_dark_version = '3.1.11'
jupyterlab_theme_dark_url = "https://unpkg.com/@jupyterlab/theme-dark-extension@%s/style/variables.css" % jupyterlab_theme_dark_version

template_css_urls = {
    'lab': [(jupyterlab_css_url, 'index.css'), (jupyterlab_theme_light_url, 'theme-light.css'), (jupyterlab_theme_dark_url, 'theme-dark.css')],
    'classic': [(notebook_css_url, 'style.css')]
}


class FetchCSS(Command):
    description = "Fetch CSS from CDN"
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def _download(self, url):
        try:
            return urlopen(url).read()
        except Exception as e:
            if 'ssl' in str(e).lower():
                try:
                    import pycurl
                except ImportError:
                    print("Failed, try again after installing PycURL with `pip install pycurl` to avoid outdated SSL.", file=sys.stderr)
                    raise e
                else:
                    print("Failed, trying again with PycURL to avoid outdated SSL.", file=sys.stderr)
                    return self._download_pycurl(url)
            raise e

    def _download_pycurl(self, url):
        """Download CSS with pycurl, in case of old SSL (e.g. Python < 2.7.9)."""
        import pycurl
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        buf = BytesIO()
        c.setopt(c.WRITEDATA, buf)
        c.perform()
        return buf.getvalue()

    def run(self):
        for template_name, resources in template_css_urls.items():
            for url, filename in resources:
                directory = os.path.join('share', 'jupyter', 'nbconvert', 'templates', template_name, 'static')
                dest = os.path.join(directory, filename)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if not os.path.exists('.git') and os.path.exists(dest):
                    # not running from git, nothing to do
                    return
                print("Downloading CSS: %s" % url)
                try:
                    css = self._download(url)
                except Exception as e:
                    msg = "Failed to download css from %s: %s" % (url, e)
                    print(msg, file=sys.stderr)
                    if os.path.exists(dest):
                        print("Already have CSS: %s, moving on." % dest)
                    else:
                        raise OSError("Need CSS to proceed.")
                    return

                with open(dest, 'wb') as f:
                    f.write(css)
                print("Downloaded Notebook CSS to %s" % dest)

        # update package data in case this created new files
        self.distribution.data_files = get_data_files()
        update_package_data(self.distribution)

cmdclass = {'css': FetchCSS}


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """
    def run(self):
        sys.exit("Aborting implicit building of eggs. Use `pip install .` to install from source.")

def css_first(command):
    class CSSFirst(command):
        def run(self):
            self.distribution.run_command('css')
            return command.run(self)
    return CSSFirst

cmdclass['build'] = css_first(build)
cmdclass['sdist'] = css_first(sdist)
cmdclass['develop'] = css_first(develop)
cmdclass['bdist_egg'] = bdist_egg if 'bdist_egg' in sys.argv else bdist_egg_disabled

for d, _, _ in os.walk(pjoin(pkg_root, 'templates')):
    g = pjoin(d[len(pkg_root)+1:], '*.*')
    package_data['nbconvert'].append(g)

version_ns = {}
with open(pjoin(here, name, '_version.py')) as f:
    exec(f.read(), {}, version_ns)

with io.open(pjoin(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def update_package_data(distribution):
    """update package_data to catch changes during setup"""
    build_py = distribution.get_command_obj('build_py')
    # distribution.package_data = find_package_data()
    # re-init build_py options which load package_data
    build_py.finalize_options()


def get_data_files():
    # Add all the templates
    data_files = []
    for (dirpath, dirnames, filenames) in os.walk('share/jupyter/nbconvert/templates/'):
        if filenames:
            data_files.append((dirpath, [os.path.join(dirpath, filename) for filename in filenames]))
    return data_files


setup_args = dict(
    name            = name,
    description     = "Converting Jupyter Notebooks",
    long_description_content_type   = 'text/markdown',
    version         = version_ns['__version__'],
    packages        = packages,
    long_description= long_description,
    package_data    = package_data,
    data_files      = get_data_files(),
    cmdclass        = cmdclass,
    python_requires = '>=3.7',
    author          = 'Jupyter Development Team',
    author_email    = 'jupyter@googlegroups.com',
    url             = 'https://jupyter.org',
    project_urls={
        'Documentation': 'https://nbconvert.readthedocs.io/en/latest/',
        'Funding'      : 'https://numfocus.org/',
        'Source'       : 'https://github.com/jupyter/nbconvert',
        'Tracker'      : 'https://github.com/jupyter/nbconvert/issues',
    },
    license         = 'BSD',
    platforms       = "Linux, Mac OS X, Windows",
    keywords        = ['Interactive', 'Interpreter', 'Shell', 'Web'],
    classifiers     = [
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)

setup_args['install_requires'] = [
    'mistune>=0.8.1,<2',
    'jinja2>=2.4',
    'pygments>=2.4.1',
    'jupyterlab_pygments',
    'traitlets>=5.0',
    'jupyter_core',
    'nbformat>=4.4',
    'entrypoints>=0.2.2',
    'bleach',
    'pandocfilters>=1.4.1',
    'testpath',
    'defusedxml',
    'nbclient>=0.5.0,<0.6.0'
]

pyppeteer_req = 'pyppeteer==0.2.6'

extra_requirements = {
    'test': [
        'pytest',
        'pytest-cov',
        'pytest-dependency',
        'ipykernel',
        'ipywidgets>=7',
        pyppeteer_req,
    ],
    'serve': [
        'tornado>=4.0'
    ],
    'webpdf': [
        pyppeteer_req
    ],
    'docs': [
        'sphinx>=1.5.1',
        'sphinx_rtd_theme',
        'nbsphinx>=0.2.12',
        'ipython',
    ],
}

extra_requirements['all'] = sum(extra_requirements.values(), [])
setup_args['extras_require'] = extra_requirements

setup_args['entry_points'] = {
    'console_scripts': [
        'jupyter-nbconvert = nbconvert.nbconvertapp:main',
        'jupyter-dejavu = nbconvert.nbconvertapp:dejavu_main',
    ],
    "nbconvert.exporters" : [
        'custom=nbconvert.exporters:TemplateExporter',
        'html=nbconvert.exporters:HTMLExporter',
        'slides=nbconvert.exporters:SlidesExporter',
        'latex=nbconvert.exporters:LatexExporter',
        'pdf=nbconvert.exporters:PDFExporter',
        'webpdf=nbconvert.exporters:WebPDFExporter',
        'markdown=nbconvert.exporters:MarkdownExporter',
        'python=nbconvert.exporters:PythonExporter',
        'rst=nbconvert.exporters:RSTExporter',
        'notebook=nbconvert.exporters:NotebookExporter',
        'asciidoc=nbconvert.exporters:ASCIIDocExporter',
        'script=nbconvert.exporters:ScriptExporter'],
}

if __name__ == '__main__':
    setup(**setup_args)
