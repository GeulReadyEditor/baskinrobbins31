<template>
	<div>
		<v-flex id="changepass-margin" class="mx-auto">
			<div align="center" class="spacing">
				<label id="title">비밀번호 변경</label>
			</div>
			<!-- 이메일 -->
			<div class="spacing-pass">
				<div style="margin-bottom:20px;">
					<label class="subtitle">이메일</label>
				</div>
				<div>
					<v-text-field :value="this.signedIn? this.email: ''" disabled></v-text-field>
				</div>
			</div>
			<v-divider class="divider-pass"/>
			<!-- 기존 비밀번호 -->
			<div class="spacing-pass">
				<div style="margin-bottom:20px;">
					<label class="subtitle">기존 비밀번호</label>
				</div>
				<div>
					<v-text-field id="cur-pass" class="cur-pass"
						v-model="curPass"
						:value="curPass"
						:append-icon="show2 ? 'mdi-eye' : 'mdi-eye-off'"
						:type="show2 ? 'text' : 'password'"
						color="#00d5aa"
						name="input-10-1"
						dense
						single-line
						outlined
						@click:append="show2 = !show2"
						/>
				</div>
			</div>
			<v-divider class="divider-pass"/>
			<!-- 새 확인 -->
			<div class="spacing-pass">
				<div style="margin-bottom:20px;">
					<label class="subtitle">새 비밀번호</label>
				</div>
				<div>
					<v-text-field id="new-pass" class="new-pass"
						v-model="newPass"
						:append-icon="show3 ? 'mdi-eye' : 'mdi-eye-off'"
						:type="show3 ? 'text' : 'password'"
						color="#00d5aa"
						name="input-10-1"
						dense
						single-line
						outlined
						@keyup.enter="changePass"
						@click:append="show3 = !show3"/>
				</div>
			</div>
			<v-divider class="divider-pass"/>
			<!-- cancel and change buttons -->
			<div align="center">
				<v-btn id="cancel-btn"
					rounded
					outlined
					color="grey"
					href="/setting">
					취소
				</v-btn>&nbsp;&nbsp;
				<v-btn id="save-btn"
					rounded
					outlined
					color="rgb(0, 213, 170)"
					@click="changePass">
					변경
				</v-btn>
			</div>
		</v-flex>
	</div>
</template>

<script lang="ts">
	import Vue from 'vue'
	import { Auth } from 'aws-amplify';

	export default Vue.extend({
		name: "ChangePassword",
		methods:{
			changePass(){
				Auth.currentAuthenticatedUser()
					.then(user => {
						return Auth.changePassword(user, this.curPass, this.newPass);
					})
					.then(data => {
						if(data == "SUCCESS"){
							Auth.signOut()
								.then(user => {
									this.$store.commit('changeSignedInState', user);
									this.$store.commit('setAccessToken', '');
									console.log("logout")
									localStorage.clear()
									window.open("/","_self");  
								})
								.catch(err => console.log(err))
								}
						alert("성공적으로 비밀번호를 변경하였습니다.")
					})
					.catch(err => console.log(err));
			},

		},
		data: () => ({
			show1: false,
			show2: false,
			show3: false,
			curPass: '',
			newPass: '',
			newPassChk: '',
			signedIn: false,
			email: '',
			rules: {
				required: (value: string) => !!value || '필수 입력사항입니다.',
				match: (v:string) => v === (document.getElementById('new-pass') as HTMLTextAreaElement).value || '비밀번호가 일치하지 않습니다.'
			},
		}),
		mounted() {
			this.signedIn = this.$store.state.user.signedIn
			this.email = this.$store.state.user.userAccount.attributes.email
		}


	})
</script>

<style>
#changepass-margin{
	max-width:55%;
	margin-top: -15px;
}
.spacing-pass{
	margin-bottom:30px;
	margin-left:20%;
	margin-right:20%;
}
.subtitle{
	color:grey;
	font-size:0.75em;
}
.divider-pass{
	margin-top:-15px;
	margin-bottom:30px;
}
#current-pass,
#new-pass,
#new-pass-check{
	font-size:1.2em !important;
}
.current-pass .v-input__slot,
.new-pass .v-input__slot,
.new-pass-check .v-input__slot{
	margin-bottom:3px !important;
}
.current-pass .v-messages__message,
.new-pass .v-messages__message,
.new-pass-check .v-messages__message{
	font-size:0.95em;
}
#cancel-btn:hover{
	color:#757575 !important;
}
#save-btn:hover{
	color:#02bf99 !important;
}
</style>