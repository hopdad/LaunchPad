![LaunchPad]([image-url](https://github.com/hopdad/LaunchPad/blob/main/logo.jpg))

A web-based tool built with Streamlit to help manage daily shipping dock operations at a logistics/warehouse environment (specifically tailored for Meijer-style workflows).

It allows users to:
- Enter cube data manually, via CSV upload, or via scanned images/screenshots/PDFs using OCR
- Calculate totals, pallet counts, trailer projections (~1600 cube = full trailer)
- Plan peddle runs (multi-stop deliveries) with manual or auto-suggested grouping
- Track actual trailers used vs. projections for future refinement
- Save data to a local SQLite database for historical tracking and future analytics

Features include role-based access (clerks vs. admins/managers), multi-file uploads, OpenCV-preprocessed OCR (EasyOCR or Tesseract), and PDF/Excel exports.

## Features

- User authentication with roles (clerks see most features; admins also see History tab)
- Data entry: manual, CSV import, or OCR from scanned sheets/images/PDFs
- Automatic peddle run suggestions (greedy bin-packing)
- Actuals tracking (enter real trailers used vs. projected)
- Summary dashboard with totals, pallet count, trailer goal/probable
- Export to PDF and Excel
- SQLite database for persistent storage (projections + actuals)
- Dark mode theme with Meijer-inspired red/blue accents

## Requirements

Python 3.8+  
Recommended: Python 3.10 or 3.11

## Installation
