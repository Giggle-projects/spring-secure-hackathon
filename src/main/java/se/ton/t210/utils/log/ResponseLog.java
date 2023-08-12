package se.ton.t210.utils.log;

import lombok.Getter;
import org.springframework.http.HttpStatus;
import org.springframework.web.util.ContentCachingResponseWrapper;
import org.springframework.web.util.WebUtils;

import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@Getter
public class ResponseLog {

    private static final String LOG_FORMAT = "[HTTP_RES] %s";

    private final HttpStatus status;

    public ResponseLog(HttpServletResponse response) {
        this.status = HttpStatus.resolve(response.getStatus());
    }

    public static String asString(ContentCachingResponseWrapper responseWrapper) {
        return new ResponseLog(responseWrapper).asLog();
    }

    public String asLog() {
        return String.format(LOG_FORMAT, status);
    }
}
