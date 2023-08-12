package se.ton.t210.service.mail;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import se.ton.t210.service.mail.form.MailForm;

@ConditionalOnProperty(value = "auth.mail.enable.mode", havingValue = "false", matchIfMissing = true)
@Component
public class LogMailServiceImpl implements MailServiceInterface {

    private static final Logger LOGGER = LoggerFactory.getLogger(LogMailServiceImpl.class);

    @Value("${auth.mail.fromAddr:cspft@gmail.com}")
    private String fromAddr;

    @Override
    public void sendMail(String userAddr, MailForm form, String fromAddr) {
        LOGGER.info("from : " + fromAddr + "\n");
        LOGGER.info("title : " + form.title() + "\n");
        LOGGER.info("content : " + form.body() + "\n");
        LOGGER.info("to : " + userAddr);
    }

    @Override
    public void sendMail(String userAddr, MailForm form) {
        sendMail(userAddr, form, fromAddr);
    }
}
