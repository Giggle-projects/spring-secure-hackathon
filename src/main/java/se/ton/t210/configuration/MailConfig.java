package se.ton.t210.configuration;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.mail.javamail.JavaMailSender;
import se.ton.t210.service.mail.JavaMailServiceImpl;
import se.ton.t210.service.mail.LogMailServiceImpl;
import se.ton.t210.service.mail.MailServiceInterface;

@Configuration
public class MailConfig {

    @Value("${auth.mail.mode:false}")
    boolean mailEnable;

    @Autowired
    private JavaMailSender javaMailSender;

    @Bean
    public MailServiceInterface mailServiceInterface() {
        if (mailEnable) {
            return new JavaMailServiceImpl(javaMailSender);
        }
        return new LogMailServiceImpl();
    }
}
