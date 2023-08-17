package se.ton.t210.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import se.ton.t210.configuration.annotation.LoginMember;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.LoginMemberInfo;

@Controller
public class PageController {

    @GetMapping("/admin")
    public String admin() {
        return "redirect:/html/admin-access.html";
    }

    @GetMapping("/html/application-information")
    public String information(@LoginMember LoginMemberInfo memberInfo) {
        if(memberInfo.getApplicationType() == ApplicationType.PoliceOfficerMale) {
            return "redirect:/html/application-information1.html";
        }
        if(memberInfo.getApplicationType() == ApplicationType.PoliceOfficerFemale) {
            return "redirect:/html/application-information2.html";
        }
        if(memberInfo.getApplicationType() == ApplicationType.FireOfficerMale) {
            return "redirect:/html/application-information3.html";
        }
        if(memberInfo.getApplicationType() == ApplicationType.FireOfficerFemale) {
            return "redirect:/html/application-information4.html";
        }
        if(memberInfo.getApplicationType() == ApplicationType.CorrectionalOfficerFemale) {
            return "redirect:/html/application-information5.html";
        }
        if(memberInfo.getApplicationType() == ApplicationType.CorrectionalOfficerMale) {
            return "redirect:/html/application-information6.html";
        }
        return "/html/application-information1.html";
    }
}
