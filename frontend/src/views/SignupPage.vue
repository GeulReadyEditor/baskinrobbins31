<template>
    <div style="margin-top:0px;">
        <div id="signup_background">
            <div id="signup_content">
                <div id="textArea"><br></div>
                <div class="signup">
                    <v-row no-gutters>
                        <v-col cols="12" md="7">
                            <!-- Signup/Login Intro -->
                            <div id="signup_intro">
                                <h1>당신의 생각을 <br>ONEGO로 펼쳐보세요.</h1>
                                <h4>AI 글쓰기 플랫폼 ONEGO</h4>
                            </div>
                        </v-col>
                        
                        <v-col cols="6" md="5">
                            <div id="signup_area">
                                <v-card class="mx-auto" max-width="400">
                                    <SignUp :info="info" v-if="info.nickname==''"/>
                                    <EmailValidation :info="info" :email="email" v-if="info.nickname!=''"/>
                                </v-card>
                            </div>
                        </v-col>
                    </v-row>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
    import Vue from 'vue'
    import SignUp from '../components/account/SignUp.vue'
    import EmailValidation from '../components/account/EmailValidation.vue'

  export default Vue.extend({
        
		data: () => ({
            info: {
                email: '',
                nickname: '',
                signedIn: false
            },
            email: ''
		}),
		components:{
            SignUp,
            EmailValidation
		},
	})
</script>
<style>
    .v-card__subtitle, .v-card__text, .v-card__title {
        padding: 30px;
    }
    /* .v-application .ma-2 {
        margin: 0px !important;
    } */
    .btn-padding-0{
        margin: 0px !important;
    }
    v-text-field{
        width: 50;
    }
    #signup_background {
        margin-top: 60px;
        background-size: 100%;
        overflow: hidden;
    }
    #signup_background:before {
        content: ' ';
        position: absolute;
        width: 100%;
        height: 100%;
        z-index: 1;
        opacity: 0.6;
        background-size: 100%;
        background-image: url('../assets/images/writer_1.jpg');
        background-repeat: no-repeat;
        -ms-background-size: cover;
        -o-background-size: cover;
        -moz-background-size: cover;
        -webkit-background-size: cover;
        background-size: cover;
    }

    #signup_content{
        z-index: 2;
        position: relative;
        color: #000;
    }
    
    #signup_area{
        /* margin-top: 130px; */
        margin-right: 180px;
    }

	#signup_intro{
        padding: 50px;
    }

    #signup_intro h1 {
        font-size: 40px;
    }
    .auth-container {
        display: var(--container-display);
        justify-content: 0;
        align-items: var(--container-align);
        min-height: 0;
    }   
</style>