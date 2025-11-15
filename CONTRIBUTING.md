# Contributing

`QuaRT` is an open-source software with the core design principle of enabling exploration of quantum algorithms for radiation transport. As such, we welcome contributions of many forms. This file describes how you can contribute to `QuaRT` and what guidelines we have in place. Any issues which are not outlined in this file are left to maintainer discretion.

## Installation

If you want to contribute code or general changes to the codebase, the preferred method is to fork `QuaRT`, clone locally and edit, and then merge through a Pull Request. Branches should be named in all-lowercase with dashes (`-`) separating segments of the name, i.e. they must satisfy the regular expression `^[a-z-]+$`.

The main branch of `QuaRT` (from which releases are built) can be cloned by running the command
```bash
git clone https://github.com/RasmitDevkota/QuaRT.git
```

We strongly urge contributors (and users!) to use virtual environments when using `QuaRT` because we do not always rely on the latest versions of all dependencies. A `virtualenv` stored in the `.quart` directory (for example) can be created by running the command
```bash
python -m virtualenv .quart
```

The package can then be installed by navigating inside the repository directory and running the command
```bash
python -m pip install -e .
```

## Documentation

Although it is not strictly necessary for all contributions, we request that contributions are accompanied with documentation. Pull requests may be deferred or denied due to lack of documentation, especially if user contributions are highly-nontrivial.

## Testing

Our test suite is a work-in-progress - we welcome contributions to this module. Test cases are the preferred approach to testing code.

## Development Cycle

The `main` branch is used as the main thread of features which are ready for the next official release (i.e. it is a "nightly" version). The `stable/*` branches are derived from `main` and used to generate official releases. It is only updated upon a release, potentially with minor exceptions.

The development cycle typically takes the following form:

1. Contributor opens Issue or discusses a potential contribution with a Maintainer.
2. Contributor starts a new branch or fork, appropriately-named, for editing.
3. Contributor ideally documents and tests all edits.
4. Contributor opens Pull Request once branch is ready to be merged and requests review from Maintainer.
    - In some cases, the Contibutor may choose to open a Draft Pull Request to discuss development prior to formal review.
5. Maintainer reviews code and provide any necessary feedback.
6. Contributor makes any necessary changes.
7. Maintainer eventually approves Pull Request and merges changes to `main`.
8. Prior to the next applicable release, Maintainer merges changes to `stable/*`.

## Use of AI Tools

We strongly discourage the use of AI in code due to the scientific nature of this project and AI's susceptibility to error, but cannot moderate against its use. If you use AI in any way when contributing to `QuaRT`, please ensure that:

- You understand all generated code and underlying tools and methods in their entirety, such that it can be documented and fully explained during review or in the future.
- You explicitly identify and describe all uses of AI in each relevant contribution.
- Your use of AI does not violate any other licenses, guidelines, policies, or other conditions, within or outside of `QuaRT`.

Any contribution which appears to be heavily-AI generated with little user input or review is subject to be rejected from merges and removed upon maintainer discretion.

