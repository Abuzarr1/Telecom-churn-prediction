CREATE TABLE IF NOT EXISTS customers (
    customer_id  SERIAL PRIMARY KEY,
    name         VARCHAR(150),
    email        VARCHAR(150) UNIQUE,
    country      VARCHAR(100),
    plan_type    VARCHAR(50),
    signup_date  DATE NOT NULL,
    churned      BOOLEAN DEFAULT FALSE,
    churn_date   DATE
);

CREATE TABLE IF NOT EXISTS subscriptions (
    sub_id       SERIAL PRIMARY KEY,
    customer_id  INT REFERENCES customers(customer_id),
    plan_name    VARCHAR(100),
    monthly_fee  NUMERIC(10,2),
    start_date   DATE,
    end_date     DATE,
    status       VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id   SERIAL PRIMARY KEY,
    sub_id       INT REFERENCES subscriptions(sub_id),
    payment_date DATE,
    amount       NUMERIC(10,2),
    status       VARCHAR(50),
    days_late    INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_events (
    event_id             SERIAL PRIMARY KEY,
    customer_id          INT REFERENCES customers(customer_id),
    event_type           VARCHAR(100),
    event_ts             TIMESTAMP,
    session_duration_sec INT
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id       SERIAL PRIMARY KEY,
    customer_id     INT REFERENCES customers(customer_id),
    created_at      DATE,
    category        VARCHAR(100),
    severity        VARCHAR(50),
    days_to_resolve INT
);

CREATE INDEX IF NOT EXISTS idx_usage_customer ON usage_events(customer_id, event_ts);
CREATE INDEX IF NOT EXISTS idx_payments_sub   ON payments(sub_id, payment_date);
CREATE INDEX IF NOT EXISTS idx_tickets_cust   ON support_tickets(customer_id, created_at);
