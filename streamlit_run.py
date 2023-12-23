import re
import sys

import streamlit.web.bootstrap


def main():
    # pylint: disable=E1101,W0212

    cmdline = " ".join(sys.argv)
    port = re.findall(r"--port=(\d+)", cmdline)
    port = int(port[0]) if port else 8501

    flag_options = {
        "server.port": port,
        "server.headless": True,
        "global.developmentMode": False,
        "server.enableCORS": False,
        "server.enableXsrfProtection": False,
        "browser.serverAddress": "localhost",
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
