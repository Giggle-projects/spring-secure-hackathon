package se.ton.t210.configuration.http;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletRequestWrapper;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

public class XssCleanHttpRequestWrapper extends HttpServletRequestWrapper {

    private static final Logger LOGGER = LoggerFactory.getLogger(XssCleanHttpRequestWrapper.class);

    private static final Map<String, String> FILTER_CHARACTERS = Map.of(
        "<", "& lt;",
        ">", "& gt;",
        "\\(", "& #40;",
        "\\)", "& #41;",
        "'", "& #39;",
        "eval\\((.*)\\)", "",
        "[\\\"\\'][\\s]*javascript:(.*)[\\\"\\']", "\"\"",
        "script", ""
    );

    public XssCleanHttpRequestWrapper(HttpServletRequest request) {
        super(request);
    }

    public String getParameter(String parameterKey) {
        return cleanXSS(super.getParameter(parameterKey));
    }

    public String[] getParameterValues(String parameterKey) {
        final String[] parameterValues = super.getParameterValues(parameterKey);
        if (parameterValues == null) {
            return null;
        }
        return Arrays.stream(parameterValues)
            .map(XssCleanHttpRequestWrapper::cleanXSS)
            .toArray(String[]::new);
    }

    public String getHeader(String name) {
        return cleanXSS(super.getHeader(name));
    }

    private static String cleanXSS(String value) {
        if (value == null) {
            return null;
        }
        final boolean needToClean = FILTER_CHARACTERS.keySet().stream()
            .anyMatch(value::contains);
        if (needToClean) {
            LOGGER.info("XssCleanFilter before : " + value);
            for (String blockChar : FILTER_CHARACTERS.keySet()) {
                value = value.replaceAll(blockChar, FILTER_CHARACTERS.get(blockChar));
            }
            LOGGER.info("XssCleanFilter after : " + value);
        }
        return value;
    }
}
