package se.ton.t210.domain;

import se.ton.t210.domain.type.ApplicationType;

import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;

@StaticMetamodel(Member.class)
public class Member_ {
    public static volatile SingularAttribute<Member, Long> id;
    public static volatile SingularAttribute<Member, String> name;
    public static volatile SingularAttribute<Member, String> email;
    public static volatile SingularAttribute<Member, String> password;
    public static volatile SingularAttribute<Member, ApplicationType> applicationType;
}
