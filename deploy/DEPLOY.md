# Deploy systemd services

Copy a `.service` file and enable it:

```bash
sudo cp deploy/sentinel-carry.service           /etc/systemd/system/
sudo cp deploy/sentinel-carry-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel-carry sentinel-carry-dashboard
sudo systemctl status sentinel-carry sentinel-carry-dashboard --no-pager
```

Ports: **funding 8787** · champion 8788 · **consensus 8789** (took carry's port when carry was retired) · trend 8790
Remember to open the GCP firewall for any new port.

---

## Funding Harvest — replaces the retired Grid (takes port 8787, already open)

Grid was retired: its apparent edge was a fill illusion (adverse selection), and it lost money in
every honest backtest. The **funding harvest** book — delta-neutral, short persistently positive
funding against a hedge — takes its port (8787), so **no firewall change is needed**.

```bash
# 1. pull latest
cd ~/sentinel && git fetch origin && git reset --hard origin/main

# 2. STOP + DISABLE anything still serving 8787 from a retired book
systemctl list-units 'sentinel*' --no-pager
# sudo systemctl disable --now sentinel-grid sentinel-grid-dashboard

# 3. install + start the four books
for u in funding champion-loop champion-dashboard carry carry-dashboard \
         trend trend-dashboard funding-dashboard; do
  sudo cp deploy/sentinel-$u.service /etc/systemd/system/ 2>/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel-funding sentinel-funding-dashboard \
     sentinel-champion-loop sentinel-champion-dashboard \
     sentinel-carry sentinel-carry-dashboard \
     sentinel-trend sentinel-trend-dashboard
systemctl list-units --type=service 'sentinel*' --no-pager --plain
```

Live lineup: **funding** (8787) · **champion** (8788) · **carry** (8789) · **trend** (8790).
All PAPER. Funding accrues hourly but only re-picks its basket every 3 days, so most ticks log
`hold ... fee 0.00` — that is the turnover control working, not a stall.
