please use subagents to find the 4 simplest classes in each              │
│   mocks/*_mocks.py file (that aren't just                                  │
│       def __init__(self, *args, **kwargs):                                 │
│           super().__init__(*args, **kwargs)) and have each agent return    │
│   the class definition to you. then, please think hard about overlaps      │
│   among the classes and propose abstract classes from which many can       │
│   inherit. the proposed classes should inherit from ABC and Mock. add      │
│   these proposed clases, along with the name of the proposed file that     │
│   would contain each, as well as the names of the files and classes in     │
│   mocks/ that should inherit from each, as new items at the bottom of      │
│   TODO.md. remember that the refactored versions of the simple mock        │
│   classes can inherit from multiple parent classes and factor this into    │
│   your reasoning. do not implement any of the proposed changes, simply     │
│   add them to the TODO list.   