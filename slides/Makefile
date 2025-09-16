# Build PDFs from my .typ (Typst) files
#
# Requirements:
# * Typst, see [the installation
#   instructions](https://github.com/typst/typst?tab=readme-ov-file#installation)
# * Libertinus fonts, from https://github.com/alerque/libertinus

.PHONY: pdf
pdf:
	typst compile -f pdf slides.typ

.PHONY: slides
slides: pdf
	open slides.pdf

.PHONY: clean
clean:
	rm -f slides.pdf

.PHONY: help
help:
	@echo "make pdf           create slides.pdf"
	@echo "make slides        create slides.pdf from slides.typ and open"
	@echo "make clean         delete slides.pdf"
	@echo "make help          gives this help text"
