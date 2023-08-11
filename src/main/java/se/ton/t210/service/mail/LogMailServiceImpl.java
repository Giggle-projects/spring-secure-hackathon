package se.ton.t210.service.mail;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import se.ton.t210.dto.Email;

@Component
public class LogMailServiceImpl implements MailServiceInterface {

    @Value("${auth.mail.fromAddress:cspft@gmail.com}")
    private String fromAddress;

    private static final Logger LOGGER = LoggerFactory.getLogger(LogMailServiceImpl.class);

    @Override
    public void sendMail(Email email) {
        sendMail(email, fromAddress);
    }

    @Override
    public void sendMail(Email email, String fromAddress) {
        LOGGER.info("from : " + fromAddress + "\n");
        LOGGER.info("title : " + email.getTitle() + "\n");
        LOGGER.info("content : " + email.getContent() + "\n");
        LOGGER.info("to : " + email.getToAddress());
    }
}
