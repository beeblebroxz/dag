#!/usr/bin/env python3
"""
Calculator Example - DAG UI Binding Demo

A simple calculator that demonstrates reactive UI bindings.
The Sum and Product labels automatically update when you
change the input values.

Run with: python examples/calculator.py
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dag
from dag.ui import DagApp, BoundEntry, BoundLabel, CellInput, CellDisplay


class Calculator(dag.Model):
    """A simple calculator model with reactive computed functions."""

    @dag.computed(dag.Input)
    def A(self):
        """First input value."""
        return 0.0

    @dag.computed(dag.Input)
    def B(self):
        """Second input value."""
        return 0.0

    @dag.computed
    def Sum(self):
        """Sum of A and B."""
        return self.A() + self.B()

    @dag.computed
    def Difference(self):
        """Difference of A and B."""
        return self.A() - self.B()

    @dag.computed
    def Product(self):
        """Product of A and B."""
        return self.A() * self.B()

    @dag.computed
    def Quotient(self):
        """Quotient of A and B (handles division by zero)."""
        b = self.B()
        if b == 0:
            return float('inf') if self.A() >= 0 else float('-inf')
        return self.A() / b


def main():
    # Reset DAG state
    dag.reset()

    # Create the application
    app = DagApp("DAG Calculator", width=350, height=250)

    # Create the calculator model
    calc = Calculator()

    # Configure grid
    app.root.columnconfigure(1, weight=1)

    # Style
    style = ttk.Style()
    style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))

    # Header
    header = ttk.Label(
        app.root,
        text="DAG Reactive Calculator",
        style='Header.TLabel'
    )
    header.grid(row=0, column=0, columnspan=2, pady=10)

    # Inputs section
    ttk.Label(app.root, text="Inputs", font=('Helvetica', 10, 'bold')).grid(
        row=1, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 5)
    )

    # Input A
    tk.Label(app.root, text="A:", width=10, anchor='e').grid(row=2, column=0, padx=5, pady=2)
    entry_a = BoundEntry(app.root, cell=calc.A, app=app, width=15)
    entry_a.grid(row=2, column=1, sticky='w', padx=5, pady=2)

    # Input B
    tk.Label(app.root, text="B:", width=10, anchor='e').grid(row=3, column=0, padx=5, pady=2)
    entry_b = BoundEntry(app.root, cell=calc.B, app=app, width=15)
    entry_b.grid(row=3, column=1, sticky='w', padx=5, pady=2)

    # Outputs section
    ttk.Label(app.root, text="Results", font=('Helvetica', 10, 'bold')).grid(
        row=4, column=0, columnspan=2, sticky='w', padx=10, pady=(15, 5)
    )

    # Sum
    tk.Label(app.root, text="A + B =", width=10, anchor='e').grid(row=5, column=0, padx=5, pady=2)
    label_sum = BoundLabel(
        app.root, cell=calc.Sum, app=app,
        width=15, anchor='w', relief='sunken', bg='white'
    )
    label_sum.grid(row=5, column=1, sticky='w', padx=5, pady=2)

    # Difference
    tk.Label(app.root, text="A - B =", width=10, anchor='e').grid(row=6, column=0, padx=5, pady=2)
    label_diff = BoundLabel(
        app.root, cell=calc.Difference, app=app,
        width=15, anchor='w', relief='sunken', bg='white'
    )
    label_diff.grid(row=6, column=1, sticky='w', padx=5, pady=2)

    # Product
    tk.Label(app.root, text="A * B =", width=10, anchor='e').grid(row=7, column=0, padx=5, pady=2)
    label_product = BoundLabel(
        app.root, cell=calc.Product, app=app,
        width=15, anchor='w', relief='sunken', bg='white'
    )
    label_product.grid(row=7, column=1, sticky='w', padx=5, pady=2)

    # Quotient
    tk.Label(app.root, text="A / B =", width=10, anchor='e').grid(row=8, column=0, padx=5, pady=2)
    label_quotient = BoundLabel(
        app.root, cell=calc.Quotient, app=app,
        width=15, anchor='w', relief='sunken', bg='white'
    )
    label_quotient.grid(row=8, column=1, sticky='w', padx=5, pady=2)

    # Instructions
    instructions = tk.Label(
        app.root,
        text="Enter values and press Tab or Enter to update",
        font=('Helvetica', 9, 'italic'),
        fg='gray'
    )
    instructions.grid(row=9, column=0, columnspan=2, pady=15)

    # Run the application
    print("Starting DAG Calculator...")
    print("Enter values in A and B fields, then press Tab or Enter.")
    print("The results will automatically update!")
    app.run()


if __name__ == '__main__':
    main()
