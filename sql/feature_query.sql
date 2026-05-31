WITH sub_features AS (
    SELECT
        s.customer_id,
        ROUND(AVG(s.monthly_fee), 2)                          AS avg_monthly_fee,
        COUNT(*)                                               AS num_subscriptions,
        SUM(CASE WHEN s.status = 'cancelled' THEN 1 ELSE 0 END) AS num_cancellations,
        CURRENT_DATE - MIN(s.start_date)                      AS tenure_days
    FROM subscriptions s
    GROUP BY s.customer_id
),
payment_features AS (
    SELECT
        s.customer_id,
        COUNT(p.payment_id)                                                AS total_payments,
        SUM(CASE WHEN p.status = 'failed'   THEN 1 ELSE 0 END)            AS failed_payments,
        SUM(CASE WHEN p.status = 'refunded' THEN 1 ELSE 0 END)            AS refunded_payments,
        ROUND(AVG(p.days_late), 1)                                         AS avg_days_late,
        ROUND(
            100.0 * SUM(CASE WHEN p.status = 'failed' THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(p.payment_id), 0), 1
        )                                                                  AS payment_failure_pct
    FROM subscriptions s
    LEFT JOIN payments p ON s.sub_id = p.sub_id
    GROUP BY s.customer_id
)
SELECT
    c.customer_id,
    c.churned::INT                                AS churned,
    c.plan_type,
    COALESCE(sf.avg_monthly_fee,      0)          AS avg_monthly_fee,
    COALESCE(sf.num_subscriptions,    0)          AS num_subscriptions,
    COALESCE(sf.tenure_days,          0)          AS tenure_days,
    COALESCE(pf.failed_payments,      0)          AS failed_payments,
    COALESCE(pf.payment_failure_pct,  0)          AS payment_failure_pct,
    COALESCE(pf.avg_days_late,        0)          AS avg_days_late,
    COALESCE(pf.refunded_payments,    0)          AS refunded_payments,
    COALESCE(pf.total_payments,       0)          AS total_payments
FROM customers c
LEFT JOIN sub_features     sf ON c.customer_id = sf.customer_id
LEFT JOIN payment_features pf ON c.customer_id = pf.customer_id;
