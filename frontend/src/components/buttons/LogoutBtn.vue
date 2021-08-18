<template>
	<v-btn id="logout-btn"
		class="ma-2"
		rounded
		outlined
		color="#00d5aa"
		@click="logout">
		로그아웃
	</v-btn>
</template>

<script lang="ts">
	import Vue from 'vue'
	import { Auth } from 'aws-amplify';

	export default Vue.extend({
		name:"LogoutBtn",
		data: ()=>({
		}),
		methods:{
			logout(){
				// this.$emit('logout', false);
				try {
					Auth.signOut()
						.then(user => {
							this.$store.commit('changeSignedInState', user);
							this.$store.commit('setAccessToken', '');
							console.log("logout")
							localStorage.clear()
							window.open("/","_self");  
						})
						.catch(err => console.log(err))
				} catch (error) {
					console.log('error signing out: ', error);
				}
			}
		}
	})
</script>

<style>
#logout-btn:hover{
	color:#02bf99 !important;
}
</style>