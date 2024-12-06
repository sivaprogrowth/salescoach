#!/bin/bash

# Connect to MySQL
mysql -h salescoachdatabase-1.ctkqiaw8wxwq.ap-south-1.rds.amazonaws.com -P 3306 -u admin -p0GdruvGwm7I5p7umY8Gr <<EOF
SET SESSION wait_timeout = 31536000;
SET SESSION interactive_timeout = 31536000;
EOF
