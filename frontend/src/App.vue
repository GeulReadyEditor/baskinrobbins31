<template>
  <v-app >
	<Header :class="$route.meta.headerClass"/>
	<router-view id="content-margin"/>
	<Footer v-if="$route.meta.showFooter"/>
  </v-app>
</template>

<script lang="ts">
	import Vue from 'vue'
	import Header from '@/components/layout/Header.vue'
	import Footer from '@/components/layout/Footer.vue'

	export default Vue.extend({
		name: "App",
		components:{
			Header, Footer
		},
		async created(){
			await this.$store.dispatch('findUser');
			var userInfo = JSON.parse(localStorage.getItem('userInfo') || '{}')
		},
	})
</script>


<style lang ="scss">
	#app {
		font-family: 'Nanum Myeongjo', serif !important;
	}
	#content-margin{
		margin-top:100px;
		margin-bottom:400px;
	}
</style>
