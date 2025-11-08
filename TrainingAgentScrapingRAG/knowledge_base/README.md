# Knowledge Base

This directory stores curated first-aid guidance documents organized by emergency scenario.

## Structure

Each scenario should be organized as follows:
```
scenario_name/
├── adult/
│   ├── guidelines.txt  # Text instructions for adults
│   └── images/         # Associated images (PNG, JPG, etc.)
└── child/
    ├── guidelines.txt  # Text instructions for children
    └── images/         # Associated images
```

## Current Scenarios

- `cardiac_arrest/` - Cardiac arrest emergency procedures
- `choking/` - Choking emergency procedures
- `heart_attack/` - Heart attack emergency procedures
- `arrhythmias/` - Heart rhythm problems (arrhythmias) emergency procedures
- `angina/` - Angina attack first aid procedures

## Adding New Scenarios

Use the `add_new_scenario()` function from the `retrieval.knowledge_loader` module, or manually create the directory structure following the pattern above.

