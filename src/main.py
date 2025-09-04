# main.py - Single Entry Point for All Systems
"""
Vehicle Registration AI - System Launcher
=========================================

Choose between different system configurations:
1. Integrated System (Recommended) - Your existing agents + LangGraph
2. Unified System - New streamlined implementation  
3. Basic Demo - Works without OpenAI key
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

def check_system_requirements():
    """Check what's available"""
    
    requirements = {
        "database": Path("db/vehicles.db").exists(),
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "data_files": len(list(Path("data").glob("*.csv"))) > 0 if Path("data").exists() else False
    }
    
    return requirements

def show_system_status():
    """Show current system status"""
    
    reqs = check_system_requirements()
    
    status_table = Table(title="🔍 System Status Check", border_style="cyan")
    status_table.add_column("Component", style="bold")
    status_table.add_column("Status", style="bold")
    status_table.add_column("Description")
    
    # Database
    db_status = "✅ Ready" if reqs["database"] else "❌ Missing"
    db_desc = "SQLite database with vehicle data" if reqs["database"] else "Run: python src/data_ingestion.py"
    status_table.add_row("Database", db_status, db_desc)
    
    # OpenAI API Key
    key_status = "✅ Set" if reqs["openai_key"] else "⚠️  Missing"  
    key_desc = "AI features enabled" if reqs["openai_key"] else "Set OPENAI_API_KEY for AI features"
    status_table.add_row("OpenAI Key", key_status, key_desc)
    
    # Data Files
    data_status = "✅ Found" if reqs["data_files"] else "⚠️  None"
    data_desc = "CSV files available for processing" if reqs["data_files"] else "Place CSV files in data/ directory"
    status_table.add_row("Data Files", data_status, data_desc)
    
    console.print(status_table)
    return reqs

def show_system_options(requirements):
    """Show available system options"""
    
    options_table = Table(title="🚀 Available Systems", border_style="green")
    options_table.add_column("Option", style="bold", width=8)
    options_table.add_column("System", style="bold cyan", width=25)
    options_table.add_column("Description", width=40)
    options_table.add_column("Requirements", width=15)
    
    # Option 1 - Integrated (Recommended)
    req1 = "✅ Ready" if requirements["database"] else "❌ Need DB"
    options_table.add_row(
        "[bold]1[/bold]",
        "🔧 Integrated System",
        "Your existing LlamaIndex agents + LangGraph orchestration", 
        req1
    )
    
    # Option 2 - Unified  
    req2 = "✅ Ready" if (requirements["database"] and requirements["openai_key"]) else "❌ Need DB+Key"
    options_table.add_row(
        "[bold]2[/bold]", 
        "🧠 Unified System",
        "New streamlined implementation with advanced AI",
        req2
    )
    
    # Option 3 - Demo
    options_table.add_row(
        "[bold]3[/bold]",
        "🎯 Basic Demo", 
        "Works without OpenAI key, shows system structure",
        "✅ Always Ready"
    )
    
    # Option 4 - Setup
    options_table.add_row(
        "[bold]setup[/bold]",
        "⚙️  Data Setup",
        "Run data ingestion pipeline",
        "✅ Always Available"
    )
    
    console.print("\n")
    console.print(options_table)

def run_integrated_system():
    """Run the integrated system (existing agents + LangGraph)"""
    
    console.print("\n[green]🚀 Starting Integrated System...[/green]")
    console.print("[dim]Using your existing agents with LangGraph orchestration[/dim]")
    
    try:
        sys.path.append("src/agents")
        from langgraph_integration import IntegratedApp
        
        app = IntegratedApp()
        if app.initialize():
            app.run_interactive()
        else:
            console.print("[red]❌ Failed to initialize integrated system[/red]")
            
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Make sure all dependencies are installed:[/yellow]")
        console.print("[dim]pip install -r requirements.txt[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")

def run_unified_system():
    """Run the new unified system"""
    
    console.print("\n[green]🚀 Starting Unified System...[/green]")
    console.print("[dim]New streamlined implementation[/dim]")
    
    try:
        sys.path.append("src")
        from unified_system import UnifiedApp
        
        app = UnifiedApp()
        if app.initialize():
            app.run_interactive()
        else:
            console.print("[red]❌ Failed to initialize unified system[/red]")
            
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")

def run_demo():
    """Run basic demo"""
    
    console.print("\n[green]🚀 Starting Basic Demo...[/green]")
    console.print("[dim]Shows system capabilities without requiring OpenAI key[/dim]")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]❌ demo.py not found[/red]")

def run_setup():
    """Run data setup"""
    
    console.print("\n[green]🚀 Starting Data Setup...[/green]")
    
    # Check for data files
    data_path = Path("data")
    if not data_path.exists() or not list(data_path.glob("*.csv")):
        console.print("[yellow]⚠️  No CSV files found in data/ directory[/yellow]")
        console.print("Please place your CSV files in the data/ directory first.")
        return
    
    try:
        sys.path.append("src") 
        from data_ingestion import main as run_ingestion
        
        run_ingestion()
        
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Setup error: {e}[/red]")

def main():
    """Main launcher"""
    
    console.clear()
    
    # Title
    title = Text("🚀 Vehicle Registration AI System", style="bold blue")
    subtitle = Text("LlamaIndex + LangGraph + Long-term Memory", style="italic green") 
    
    console.print(Panel(title, subtitle=subtitle, border_style="cyan"))
    
    # Check system status
    console.print("\n")
    requirements = show_system_status()
    
    # Show options
    show_system_options(requirements)
    
    # Get user choice
    console.print(f"\n[bold cyan]Recommended:[/bold cyan] Option 1 (Integrated System)")
    choice = console.input("\n[bold]Select option[/bold] [dim](1, 2, 3, setup, quit)[/dim]: ").strip().lower()
    
    if choice == "1":
        if not requirements["database"]:
            console.print("\n[red]❌ Database required! Run 'setup' first.[/red]")
        else:
            run_integrated_system()
            
    elif choice == "2":
        if not requirements["database"]:
            console.print("\n[red]❌ Database required! Run 'setup' first.[/red]")
        elif not requirements["openai_key"]:
            console.print("\n[red]❌ OpenAI API key required! Set OPENAI_API_KEY.[/red]")
        else:
            run_unified_system()
            
    elif choice == "3":
        run_demo()
        
    elif choice == "setup":
        run_setup()
        
    elif choice == "quit":
        console.print("\n[green]👋 Goodbye![/green]")
        
    else:
        console.print(f"\n[red]❌ Invalid choice: {choice}[/red]")

if __name__ == "__main__":
    main()