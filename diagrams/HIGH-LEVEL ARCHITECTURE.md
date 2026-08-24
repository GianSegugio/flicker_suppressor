```text
                         INPUT RGB
                             |
                             v
                 EXIF orientation normalization
                             |
                             v
                  band-axis decision/override
                             |
             +---------------+---------------+
             |                               |
       horizontal bands                vertical bands
       process directly            rotate 90 degrees
             |                               |
             +---------------+---------------+
                             |
                             v
                    processing RGB image
                             |
             +---------------+----------------+
             |                                |
     Restormer enabled                  Restormer disabled
             |                                |
   resize to working size                     |
      +------+-------+                        |
      |              |                        |
      v              v                        |
  Y Restormer    CbCr branch                  |
      |              |                        |
 derive constrained corrections               |
      +------+-------+                        |
             |                                |
   apply to full-resolution Y/CbCr            |
             |                                |
      optional second pass                    |
             +---------------+----------------+
                             |
                             v
                    deterministic cleanup
        +--------------------+---------------------+
        |                    |                     |
  local flat filter    robust residual profile  broad cleanup
                             |
                    PWM mode may use:
            fundamental-first Auto timing,
       phase lock / multi-source surface fitting,
               optional final PWM polish
                             |
                optional orthogonal profile
                             |
                      tone restoration
                             |
                    restore orientation
                             |
                          PNG output
```

---

*Last Updated: 24 August 2026*
