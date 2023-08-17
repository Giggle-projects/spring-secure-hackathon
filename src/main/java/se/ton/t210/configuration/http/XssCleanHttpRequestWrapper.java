package se.ton.t210.configuration.http;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ReadListener;
import javax.servlet.ServletInputStream;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletRequestWrapper;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;

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

    private byte[] body;

    public XssCleanHttpRequestWrapper(HttpServletRequest request) {
        super(request);
        try {
            InputStream is = request.getInputStream();
            if (is != null) {
                StringBuilder sb = new StringBuilder();
                while (true) {
                    int data = is.read();
                    if (data < 0) {
                        break;
                    }
                    sb.append((char) data);
                }
                body = cleanXSS(sb.toString()).getBytes(StandardCharsets.ISO_8859_1);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
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

    @Override
    public ServletInputStream getInputStream() {
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(this.body);
        return new ServletInputStream() {

            @Override
            public boolean isFinished() {
                return byteArrayInputStream.available() == 0;
            }

            @Override
            public boolean isReady() {
                return true;
            }

            @Override
            public void setReadListener(ReadListener readListener) {

            }

            @Override
            public int read() {
                return byteArrayInputStream.read();
            }
        };
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
