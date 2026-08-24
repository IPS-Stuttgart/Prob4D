- Route legacy PhysTwin experiment and state pickle inputs through the existing
  restricted NumPy-only loader, reject unexpected final-data containers, and add
  an AST policy that prevents new unsafe deserialization while preserving the
  exact frozen CUT3R source-freeze-v1 exception.
