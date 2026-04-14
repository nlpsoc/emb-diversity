# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
APIDOCDIR     = ../src/embediver

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate API documentation from source code
apidoc:
	@echo "Generating API documentation from source code..."
	@sphinx-apidoc -f --no-headings -o $(SOURCEDIR) $(APIDOCDIR)
	@echo "API documentation generated!"

clean:
	@test -n "$(BUILDDIR)" || (echo "ERROR: BUILDDIR is empty" && exit 1)
	@test -d "$(BUILDDIR)" || (echo "ERROR: BUILDDIR does not exist" && exit 1)
	@test "$(BUILDDIR)" != "/" || (echo "ERROR: BUILDDIR is root" && exit 1)
	rm -rf "$(BUILDDIR)"
	mkdir -p "$(BUILDDIR)"

# Clean build artifacts and generated RST files
cleanall: clean
	@echo "Removing generated API documentation files..."
	@rm -f $(SOURCEDIR)/embediver*.rst
	@rm -f $(SOURCEDIR)/modules.rst
	@echo "All clean!"

.PHONY: help apidoc clean cleanall

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
