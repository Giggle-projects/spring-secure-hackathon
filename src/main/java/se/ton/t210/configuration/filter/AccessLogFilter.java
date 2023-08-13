package se.ton.t210.configuration.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.util.ContentCachingRequestWrapper;
import org.springframework.web.util.ContentCachingResponseWrapper;
import se.ton.t210.utils.log.RequestLog;
import se.ton.t210.utils.log.ResponseLog;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebFilter(urlPatterns = {"/api/*"})
public class AccessLogFilter implements Filter {

    private static final Logger LOGGER = LoggerFactory.getLogger(AccessLogFilter.class);

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
        responseWrapper.copyBodyToResponse();
        printLog(requestWrapper, responseWrapper);
    }

    private void printLog(ContentCachingRequestWrapper requestWrapper, ContentCachingResponseWrapper responseWrapper) {
        if (requestEnable) {
            LOGGER.info(RequestLog.asString(requestWrapper, accessTokenCookieKey));
        }
        if (responseEnable) {
            LOGGER.info(ResponseLog.asString(responseWrapper));
        }
    }
}
