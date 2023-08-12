package se.ton.t210.configuration.filter;

import java.io.IOException;
import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.util.ContentCachingRequestWrapper;
import org.springframework.web.util.ContentCachingResponseWrapper;
import se.ton.t210.utils.log.RequestLog;
import se.ton.t210.utils.log.ResponseLog;

@WebFilter(urlPatterns = {"/api/*", "/actuator/*"})
public class HttpAccessLogFilter implements Filter {

    private static final Logger LOGGER = LoggerFactory.getLogger(HttpAccessLogFilter.class);

    @Value("${auth.jwt.token.access.cookie:accessToken}")
    private String accessTokenCookieKey;

    @Value("${global.log.http.access.request.enable:true}")
    private boolean requestEnable;

    @Value("${global.log.http.access.response.enable:true}")
    private boolean responseEnable;

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        if (!requestEnable && !responseEnable) {
            chain.doFilter(request, response);
            return;
        }
        var requestWrapper = new ContentCachingRequestWrapper((HttpServletRequest) request);
        var responseWrapper = new ContentCachingResponseWrapper((HttpServletResponse) response);
        chain.doFilter(requestWrapper, responseWrapper);
        printLog(requestWrapper, responseWrapper);
    }

    private void printLog(ContentCachingRequestWrapper requestWrapper, ContentCachingResponseWrapper responseWrapper) {
        final RequestLog request = new RequestLog(requestWrapper, accessTokenCookieKey);
        final ResponseLog response = new ResponseLog(responseWrapper);
        if (requestEnable) {
            LOGGER.info(requestLog(request));
        }
        if (responseEnable) {
            LOGGER.info(responseLog(response));
        }
    }

    private String requestLog(RequestLog request) {
        return request.asLog();
    }

    private String responseLog(ResponseLog response) {
        return response.asLog();
    }
}
