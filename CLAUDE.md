# Sift

## Code rules

- No backwards compatibility and no old code anywhere in the codebase. When a feature, setting, column, format or behavior changes or goes away, delete everything that existed for the old version in the same change: database migrations and old schema snapshots, compatibility shims, reads of old data or formats, cleanup of what earlier builds left behind (notification channels, files, preferences), deprecated aliases, unused resources, tests of removed behavior, and comments or docs that describe how things used to work. The code describes only the current design.
- Database schema changes bump the Room version and nothing else: there are no migrations and no exported schemas. Room's destructive fallback recreates the database, and Sift rebuilds it from the system SMS store (the user's marks, learning entries and labels are lost). Tell the user before installing a build with a new database version on their phone.
