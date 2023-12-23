import re
import sys

import streamlit.web.bootstrap


def main():
    # pylint: disable=E1101,W0212

    cmdline = " ".join(sys.argv)

    flag_options = {
        "server.headless": True,
        "global.developmentMode": False,
        "server.enableCORS": True,
        "server.enableXsrfProtection": False,
        "theme.base": "dark",
        "browser.gatherUsageStats": False,
        "client.showErrorDetails": False,
    }

    streamlit.web.bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    streamlit.web.bootstrap.run(
        "page.py",
        "streamlit run",
        [],
        flag_options,
    )


if __name__ == "__main__":
    main()
