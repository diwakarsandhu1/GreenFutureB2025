# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['start_django.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data_science/*', 'data_science')
    ],
    hiddenimports=[
        'altgraph',
        'asgiref',
        'cvxopt',
        'dj-database-url',
        'django',
        'django-cors-headers',
        'djangorestframework',
        'gunicorn',
        'macholib',
        'numpy',
        'packaging',
        'pandas',
        'pillow',
        'psycopg2',
        'pyinstaller',
        'pyinstaller-hooks-contrib',
        'python-dateutil',
        'python-decouple',
        'pytz',
        'setuptools',
        'six',
        'sqlparse',
        'typing_extensions',
        'tzdata',
        'whitenoise'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='start_django',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    onefile=True,  # Enable single file output
)