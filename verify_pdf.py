"""Quick verification script to check if Ambedkar_book.pdf is in the correct format."""

import sys
import warnings
from pathlib import Path

from pypdf import PdfReader
from rich.console import Console
from rich.panel import Panel

# Suppress pypdf warnings about PDF structure (we'll test if it still works)
warnings.filterwarnings("ignore", category=UserWarning)

# Fix Windows encoding issues
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

console = Console()


def verify_pdf(pdf_path: Path) -> None:
    """Verify that the PDF can be read and text can be extracted."""
    console.rule("[bold]PDF Format Verification[/bold]")
    
    if not pdf_path.exists():
        console.print(f"[red][ERROR] PDF file not found: {pdf_path}[/red]")
        return
    
    console.print(f"[cyan]Checking PDF: {pdf_path}[/cyan]\n")
    
    try:
        # Try to read PDF - may show warnings but might still work
        console.print("[yellow]Attempting to read PDF (warnings may appear but PDF might still work)...[/yellow]\n")
        reader = PdfReader(str(pdf_path), strict=False)  # strict=False to handle minor issues
        page_count = len(reader.pages)
        
        console.print("[green][OK] PDF opened successfully[/green]")
        console.print(f"[green][OK] Total pages: {page_count}[/green]")
        console.print("[dim]Note: PDF structure warnings are common and don't always prevent text extraction[/dim]\n")
        
        if page_count == 0:
            console.print("[red][ERROR] PDF has no pages![/red]")
            return
        
        # Test first few pages
        pages_to_test = min(5, page_count)
        total_text_length = 0
        pages_with_text = 0
        
        console.print(f"[yellow]Testing text extraction from first {pages_to_test} pages...[/yellow]\n")
        
        for i in range(pages_to_test):
            page = reader.pages[i]
            text = page.extract_text() or ""
            text_length = len(text.strip())
            total_text_length += text_length
            
            if text_length > 0:
                pages_with_text += 1
                preview = text[:150].replace("\n", " ").strip()
                console.print(f"  [green]Page {i+1}:[/green] {text_length} characters")
                console.print(f"    Preview: {preview}...")
            else:
                console.print(f"  [red]Page {i+1}:[/red] No text extracted (may be image-only)")
        
        # Test middle and last pages
        if page_count > 10:
            console.print(f"\n[yellow]Testing middle and last pages...[/yellow]\n")
            for page_num in [page_count // 2, page_count - 1]:
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                text_length = len(text.strip())
                if text_length > 0:
                    console.print(f"  [green]Page {page_num+1}:[/green] {text_length} characters")
                else:
                    console.print(f"  [red]Page {page_num+1}:[/red] No text extracted")
        
        # Summary
        console.print("\n" + "=" * 60)
        console.print(f"[bold]Summary:[/bold]")
        console.print(f"  Total pages: {page_count}")
        console.print(f"  Pages with extractable text: {pages_with_text}/{pages_to_test} (tested)")
        console.print(f"  Total text extracted (tested pages): {total_text_length} characters")
        
        if pages_with_text == 0:
            console.print("\n[red][ERROR] WARNING: No text could be extracted from any tested pages![/red]")
            console.print("[yellow]This PDF may be image-only (scanned) and needs OCR.[/yellow]")
        elif pages_with_text < pages_to_test:
            console.print(f"\n[yellow][WARNING] Some pages ({pages_to_test - pages_with_text}) have no extractable text.[/yellow]")
            console.print("[yellow]This may affect processing, but the PDF should still work.[/yellow]")
        else:
            console.print("\n[green][SUCCESS] PDF format is CORRECT![/green]")
            console.print("[green]The PDF has extractable text and should work with the chunking pipeline.[/green]")
        
        # Try to extract a few sentences to verify NLTK compatibility
        if total_text_length > 0:
            console.print("\n[yellow]Testing sentence tokenization...[/yellow]")
            try:
                import nltk
                nltk.download("punkt", quiet=True)
                from nltk.tokenize import sent_tokenize
                
                # Get text from first page with content
                for i in range(min(3, page_count)):
                    page = reader.pages[i]
                    text = page.extract_text() or ""
                    if text.strip():
                        sentences = sent_tokenize(text)
                        console.print(f"  [green][OK] Sentence tokenization works[/green]")
                        console.print(f"    Found {len(sentences)} sentences on page {i+1}")
                        if sentences:
                            console.print(f"    First sentence: {sentences[0][:100]}...")
                        break
            except Exception as e:
                console.print(f"  [yellow][WARNING] Sentence tokenization test skipped: {e}[/yellow]")
        
    except Exception as e:
        console.print(f"[red][ERROR] Error reading PDF: {e}[/red]")
        console.print("[yellow]The PDF may be corrupted or in an unsupported format.[/yellow]")
        console.print("\n[bold]Possible issues:[/bold]")
        console.print("  1. PDF structure is corrupted (invalid root object)")
        console.print("  2. PDF is password-protected")
        console.print("  3. PDF format is not supported by pypdf")
        console.print("\n[yellow]Try:[/yellow]")
        console.print("  - Opening the PDF in a PDF viewer to verify it's not corrupted")
        console.print("  - Converting the PDF using a tool like Adobe Acrobat or online converters")
        console.print("  - Using a different PDF library (PyMuPDF/fitz) if pypdf fails")
        return


if __name__ == "__main__":
    pdf_path = Path("data/Ambedkar_book.pdf")
    verify_pdf(pdf_path)

