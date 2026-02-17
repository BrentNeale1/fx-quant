CREATE TABLE economic_calendar (
    event_datetime   TIMESTAMPTZ       NOT NULL,
    country          TEXT              NOT NULL,
    event_name       TEXT              NOT NULL,
    currency         TEXT,
    impact           TEXT,
    impact_numeric   INTEGER,
    actual           DOUBLE PRECISION,
    forecast         DOUBLE PRECISION,
    previous         DOUBLE PRECISION,
    revised          DOUBLE PRECISION,
    reference        TEXT,
    source           TEXT,
    last_updated     TIMESTAMPTZ       DEFAULT now(),
    PRIMARY KEY (event_datetime, country, event_name)
);

CREATE INDEX idx_ec_datetime  ON economic_calendar (event_datetime DESC);
CREATE INDEX idx_ec_currency  ON economic_calendar (currency);
CREATE INDEX idx_ec_impact    ON economic_calendar (impact_numeric);
