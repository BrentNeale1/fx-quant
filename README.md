# fx-quant

## Web Dashboard

A Flask-based web UI for managing the trading bot remotely — edit config, monitor status, view logs, and toggle the kill switch from a browser instead of SSH + manual YAML editing.

### Setup

1. **Set your dashboard password** in `config/.env`:
   ```
   DASHBOARD_PASSWORD=your-secure-password-here
   ```

2. **Build and start** both services:
   ```bash
   docker-compose build && docker-compose up -d
   ```

3. **Access via SSH tunnel** (dashboard is not exposed publicly):
   ```bash
   ssh -L 5000:localhost:5000 your-server
   ```
   Then open http://localhost:5000 in your browser.

4. **Log in** with username `admin` and the password you set in step 1.

### Dashboard Pages

- **Status** (`/`) — Current mode (paper/live), strategy, instruments, loop interval, kill switch toggle, and last 10 orders
- **Config** (`/config`) — Form-based editor for all `system.yaml` sections: instruments, granularities, strategy params, feature windows, AI settings, execution settings. Saves with backup and signals the bot to reload.
- **Logs** (`/logs`) — Tabbed tables showing order history (`logs/order_log.csv`) and AI decisions (`logs/ai_decisions.csv`), newest first

### Config Reload Flow

When you save config changes through the dashboard:

1. Dashboard backs up `system.yaml` to `system.yaml.backup`
2. Dashboard writes the updated config
3. Dashboard creates a `RELOAD_CONFIG` signal file
4. Bot checks for this file at the top of each 60s loop iteration
5. Bot reloads config, deletes the signal file, and continues with new settings

### Kill Switch

The dashboard provides activate/deactivate buttons (with confirmation prompts) that create/remove the `STOP_ALL_TRADING` file — the same mechanism the bot already uses.

### Files Added/Changed

| File | What |
|------|------|
| `src/dashboard.py` | Flask app — routes, auth, config editor, status, logs |
| `templates/base.html` | Base layout (Bootstrap 5 via CDN) |
| `templates/index.html` | Status page |
| `templates/config.html` | Config editor form |
| `templates/logs.html` | Order log + AI decisions tables |
| `docker-compose.yml` | Added `dashboard` service on port 5000 |
| `requirements.docker.txt` | Added `flask`, `flask-httpauth` |
| `config/.env` | Added `DASHBOARD_PASSWORD` |
| `src/order_executor.py` | Added `RELOAD_CONFIG` signal check in main loop |

### Verification Checklist

- [ ] `docker-compose build && docker-compose up -d` — both containers start
- [ ] http://localhost:5000 shows login prompt (via SSH tunnel)
- [ ] Status page shows current config and kill switch state
- [ ] Edit a setting (e.g. add `GBP_USD` to instruments), save
- [ ] Bot logs show `CONFIG RELOAD REQUESTED` within 60s
- [ ] Logs page shows order history and AI decisions
- [ ] Kill switch toggle works with confirmation dialog