| scenario | seed | verdict | ops | ok | fail | info | acked writes | lost | info present/absent | minority writes (blocked) | block latency | stale reads in fault | elections | failover min/med/max | converge | G1 | G2 | G3 | G5 | G7 | G8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kill-primary | 1 | PASS | 608987 | 508672 | 99894 | 421 | 356152 | 0 | 204/62 | 0 (0) | - | 0 | 2 | 0.3/0.3/0.3s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| pause-primary | 1 | PASS | 87679 | 85336 | 2125 | 218 | 60405 | 0 | 130/14 | 0 (0) | - | 16 | 2 | 33.3/33.3/33.3s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| isolate-master | 1 | PASS | 12699 | 12001 | 429 | 269 | 8376 | 0 | 98/51 | 36 (33) | 33.3/56.3/60.1s | 24 | 5 | 32.5/34.0/70.2s | 53.7s | pass | pass | pass | pass | pass | pass |
| majority-minority | 1 | PASS | 12484 | 11859 | 380 | 245 | 8299 | 0 | 69/60 | 12 (5) | 7.8/58.1/58.1s | 17 | 6 | 32.5/33.0/63.3s | 26.6s | pass | pass | pass | pass | pass | pass |
| bridge | 1 | PASS | 12681 | 11969 | 420 | 292 | 8365 | 0 | 78/104 | 0 (0) | - | 40 | 1 | - | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| asymmetric | 1 | PASS | 25706 | 24724 | 731 | 251 | 17379 | 0 | 93/61 | 51 (37) | 44.3/60.0/60.0s | 33 | 3 | 33.0/33.3/64.9s | 0.0s | pass | pass | pass | pass | pass | pass |
| flapping | 1 | PASS | 30956 | 30158 | 748 | 50 | 21340 | 0 | 20/8 | 0 (0) | - | 1 | 1 | - | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| rolling-restart | 1 | PASS | 45031 | 40086 | 4913 | 32 | 28117 | 0 | 12/3 | 0 (0) | - | 0 | 2 | 1.1/1.1/1.1s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| full-restart | 1 | PASS | 73661 | 47240 | 26299 | 122 | 33453 | 0 | 40/32 | 0 (0) | - | 0 | 4 | 4.9/4.9/4.9s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| kill-during-recovery | 1 | PASS | 705849 | 679464 | 26290 | 95 | 480798 | 0 | 31/39 | 0 (0) | - | 0 | 4 | 0.1/0.4/1.0s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| chaos | 1 | PASS | 442013 | 328197 | 113636 | 180 | 227734 | 0 | 98/28 | 29 (29) | 60.0/60.0/60.0s | 10 | 3 | 0.3/1.0/112.6s | 18.6s | pass | pass | pass | pass | pass | pass |
| kill-primary | 2 | PASS | 611576 | 509748 | 101475 | 353 | 356664 | 0 | 153/53 | 0 (0) | - | 0 | 7 | 0.2/0.3/0.3s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| pause-primary | 2 | PASS | 99998 | 97326 | 2498 | 174 | 68577 | 0 | 99/26 | 0 (0) | - | 2 | 2 | 33.1/33.1/33.1s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| isolate-master | 2 | PASS | 20730 | 19878 | 593 | 259 | 13986 | 0 | 106/64 | 0 (0) | 34.7/37.9/37.9s | 23 | 5 | 32.6/33.1/78.8s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| majority-minority | 2 | PASS | 9462 | 8817 | 348 | 297 | 6178 | 0 | 84/78 | 45 (8) | 8.5/11.1/17.1s | 17 | 5 | 33.0/105.2/105.2s | 69.3s | pass | pass | pass | pass | pass | pass |
| bridge | 2 | PASS | 9603 | 8991 | 339 | 273 | 6313 | 0 | 72/72 | 0 (0) | - | 26 | 3 | 33.4/106.3/107.9s | 63.3s | pass | pass | pass | pass | pass | pass (not exercised) |
| asymmetric | 2 | PASS | 10796 | 10118 | 361 | 317 | 7122 | 0 | 157/42 | 51 (29) | 43.1/90.3/90.3s | 10 | 6 | 33.1/33.3/33.6s | 11.6s | pass | pass | pass | pass | pass | pass |
| flapping | 2 | PASS | 52555 | 51139 | 1342 | 74 | 36016 | 0 | 56/0 | 0 (0) | - | 1 | 1 | - | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| rolling-restart | 2 | PASS | 43758 | 38805 | 4918 | 35 | 27188 | 0 | 19/5 | 0 (0) | - | 0 | 2 | 0.2/0.2/0.2s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| full-restart | 2 | PASS | 70155 | 50623 | 19358 | 174 | 35710 | 0 | 87/45 | 0 (0) | - | 0 | 6 | 0.2/3.4/3.4s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| kill-during-recovery | 2 | PASS | 701648 | 674365 | 27214 | 69 | 476625 | 0 | 34/11 | 0 (0) | - | 0 | 3 | 0.2/0.4/0.4s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| chaos | 2 | PASS | 125759 | 101814 | 23666 | 279 | 70934 | 0 | 138/58 | 0 (0) | 0.0/0.0/0.0s | 13 | 4 | 0.3/0.4/0.4s | 19.6s | pass | pass | pass | pass | pass | pass (not exercised) |
| kill-primary | 3 | PASS | 605085 | 503235 | 101451 | 399 | 352016 | 0 | 163/89 | 0 (0) | - | 0 | 2 | 0.3/0.3/0.3s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| pause-primary | 3 | PASS | 95937 | 93260 | 2434 | 243 | 65943 | 0 | 134/38 | 0 (0) | - | 3 | 3 | 32.5/32.9/32.9s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| isolate-master | 3 | PASS | 9216 | 8679 | 279 | 258 | 6066 | 0 | 160/21 | 19 (13) | 9.8/53.8/53.8s | 4 | 5 | 32.3/33.3/67.4s | 0.5s | pass | pass | pass | pass | pass | pass |
| majority-minority | 3 | PASS | 9785 | 9060 | 365 | 360 | 6309 | 0 | 116/134 | 24 (14) | 39.1/55.5/55.5s | 24 | 2 | 33.4/33.4/33.4s | 51.7s | pass | pass | pass | pass | pass | pass |
| bridge | 3 | PASS | 9454 | 8714 | 429 | 311 | 6035 | 0 | 80/100 | 0 (0) | - | 56 | 1 | - | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| asymmetric | 3 | PASS | 14081 | 13449 | 417 | 215 | 9480 | 0 | 55/65 | 30 (24) | 53.5/55.7/60.0s | 20 | 3 | 32.8/61.8/61.8s | 23.1s | pass | pass | pass | pass | pass | pass |
| flapping | 3 | PASS | 49052 | 47735 | 1228 | 89 | 33762 | 0 | 21/23 | 0 (0) | 1.7/1.7/1.7s | 12 | 1 | - | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| rolling-restart | 3 | PASS | 45495 | 40613 | 4849 | 33 | 28514 | 0 | 21/0 | 0 (0) | - | 0 | 5 | 0.3/0.4/0.4s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| full-restart | 3 | PASS | 62622 | 42924 | 19557 | 141 | 30412 | 0 | 66/37 | 0 (0) | - | 0 | 5 | 0.1/3.0/3.0s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| kill-during-recovery | 3 | PASS | 609472 | 588747 | 20680 | 45 | 416594 | 0 | 10/20 | 0 (0) | - | 0 | 2 | 0.4/0.4/0.4s | 0.0s | pass | pass | pass | pass | pass | pass (not exercised) |
| chaos | 3 | PASS | 155753 | 151810 | 3698 | 245 | 107483 | 0 | 107/51 | 9 (9) | 47.3/47.3/47.3s | 9 | 3 | 32.8/32.8/32.8s | 0.0s | pass | pass | pass | pass | pass | pass |
