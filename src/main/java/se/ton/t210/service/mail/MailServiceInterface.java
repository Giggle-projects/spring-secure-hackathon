package se.ton.t210.service.mail;

import se.ton.t210.service.mail.form.MailForm;

public interface MailServiceInterface {

    void sendMail(String userAddr, MailForm mailForm, String fromAddr);

    void sendMail(String userAddr, MailForm mailForm);
}
